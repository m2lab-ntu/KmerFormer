"""Bounded-memory labelled evaluation for independent FASTA/FASTQ pools."""
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import time

import numpy as np
import torch
from .kmer_tokenizers import build_tokenizer
from .model import create_model
from .predict import logits_for_sequences
from .reads import batches, iter_reads
from .training_utils import checkpoint_weights


class OnlineMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.topk = {k: 0 for k in (3, 5, 10) if k < num_classes}
        self.count = 0

    def update(self, labels, probabilities):
        labels = np.asarray(labels, dtype=np.int64)
        predictions = probabilities.argmax(-1)
        if labels.size and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError("Evaluation labels are outside the trained output space")
        np.add.at(self.confusion, (labels, predictions), 1)
        order = np.argsort(probabilities, axis=1, kind="mergesort")
        for k in self.topk:
            self.topk[k] += int((order[:, -k:] == labels[:, None]).any(axis=1).sum())
        self.count += len(labels)
        return predictions

    def per_class(self):
        support, predicted = self.confusion.sum(1), self.confusion.sum(0)
        correct = self.confusion.diagonal()
        precision = np.divide(correct, predicted, out=np.zeros(self.num_classes), where=predicted != 0)
        recall = np.divide(correct, support, out=np.zeros(self.num_classes), where=support != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(self.num_classes), where=(precision + recall) != 0)
        return support, predicted, precision, recall, f1

    def result(self):
        if not self.count:
            raise ValueError("Evaluation contains no matched reads")
        support, predicted, precision, recall, f1 = self.per_class()
        present, union = support > 0, (support + predicted) > 0
        result = {"micro_accuracy": float(self.confusion.trace() / self.count),
                  "balanced_accuracy": float(recall[present].mean()),
                  "num_classes": self.num_classes, "num_classes_present": int(present.sum()),
                  "num_classes_f1_zero": int((present & (f1 == 0)).sum())}
        for name, array in (("f1", f1), ("precision", precision), ("recall", recall)):
            result[name + "_macro"] = float(array[union].mean())
            result[name + "_weighted"] = float((array * support).sum() / self.count)
        result.update({f"top{k}_accuracy": hits / self.count for k, hits in self.topk.items()})
        return result


def _label_database(path, connection, task):
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("CREATE TABLE labels (read_id TEXT PRIMARY KEY, class_id INTEGER NOT NULL)")
    names, count, pending = {}, 0, []
    with open(path, newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"seq_id", f"{task}_class", f"{task}_name"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Labels TSV requires columns: {sorted(required)}")
        for row in reader:
            key, name = int(row[f"{task}_class"]), row[f"{task}_name"]
            if key < 0 or not name or (key in names and names[key] != name):
                raise ValueError("Each nonnegative class ID must have one nonempty name")
            names[key] = name
            pending.append((row["seq_id"], key))
            count += 1
            if len(pending) == 10_000:
                connection.executemany("INSERT INTO labels VALUES (?, ?)", pending)
                pending.clear()
        connection.executemany("INSERT INTO labels VALUES (?, ?)", pending)
    connection.commit()
    if not count:
        raise ValueError("Labels TSV contains no reads")
    return names


def evaluate_stream(args, cfg):
    task = cfg["data"].get("task", "genus")
    output = Path(args.output_dir or cfg["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = args.checkpoint or str(Path(cfg["output"]["dir"]) / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["model_state_dict"]
    # Every supported head exposes its output layer under one of these keys.
    output_key = next((key for key in ("head.classifier.4.weight", "head.classifier.weight") if key in state), None)
    if output_key is None:
        raise ValueError("Cannot infer checkpoint output width; unsupported classifier head")
    num_classes = int(state[output_key].shape[0])
    if args.num_classes is not None and args.num_classes != num_classes:
        raise ValueError("--num_classes disagrees with checkpoint output width")
    tokenizer = build_tokenizer(cfg)
    model_cfg = dict(cfg["model"], gradient_checkpointing=False)
    core = model_cfg.get("type", "kmerformer") in ("kmerformer", "shallow_transformer")
    if core:
        with torch.device("meta"):
            model = create_model(model_cfg, num_classes)
    else:
        model = create_model(model_cfg, num_classes)
    model.load_state_dict(checkpoint_weights(model, state), strict=True, assign=core)
    epoch = int(checkpoint.get("epoch", -1))
    del state, checkpoint
    model.to(device).eval()
    precision = args.precision
    if precision is None:
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" and cfg["training"].get("amp", True) else "fp32"
    batch_size = args.batch_size or cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"] * 2)
    max_length = cfg["data"].get("max_token_length", 128)
    metrics, forward_metrics = OnlineMetrics(num_classes), OnlineMetrics(num_classes)
    suffix = "_rc_tta" if args.rc_tta else ""
    started, inference_time = time.perf_counter(), 0.0
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output) as temporary:
        work = Path(temporary)
        with sqlite3.connect(work / "labels.sqlite") as connection:
            names = _label_database(args.test_labels, connection, task)
            arrays = {"preds": (np.int64, ()), "labels": (np.int64, ())}
            if not args.skip_save_logits:
                arrays.update({"probs": (np.float32, (num_classes,)), "logits": (np.float32, (num_classes,))})
                if args.rc_tta:
                    arrays.update({"fwd_logits": (np.float32, (num_classes,)), "rc_logits": (np.float32, (num_classes,))})
            handles = {key: (work / (key + ".bin")).open("wb") for key in arrays}
            try:
                with (work / "read_predictions.tsv").open("w", newline="") as dest:
                    writer = csv.writer(dest, delimiter="\t")
                    writer.writerow(("read_id", "true_class_id", "predicted_class_id", "taxon_name", "score"))
                    for chunk in batches(iter_reads(args.test_fasta, normalize=False), batch_size):
                        truth = []
                        for read in chunk:
                            found = connection.execute("SELECT class_id FROM labels WHERE read_id=?", (read.read_id,)).fetchone()
                            if found is None:
                                raise ValueError(f"Read {read.read_id!r} has no label; evaluation requires complete labels")
                            truth.append(found[0])
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        t0 = time.perf_counter()
                        logits, forward, reverse = logits_for_sequences(model, tokenizer, [r.sequence for r in chunk],
                            device=device, max_length=max_length, rc_tta=args.rc_tta, precision=precision,
                            kmer_preprocess=cfg["data"].get("kmer_preprocess"))
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_time += time.perf_counter() - t0
                        probabilities = logits.softmax(-1).cpu().numpy()
                        prediction = metrics.update(truth, probabilities)
                        if args.rc_tta:
                            forward_metrics.update(truth, forward.softmax(-1).cpu().numpy())
                        values = {"preds": prediction.astype(np.int64), "labels": np.asarray(truth, dtype=np.int64)}
                        if not args.skip_save_logits:
                            values.update(probs=probabilities, logits=logits.cpu().numpy())
                            if args.rc_tta:
                                values.update(fwd_logits=forward.cpu().numpy(), rc_logits=reverse.cpu().numpy())
                        for key, value in values.items():
                            handles[key].write(np.ascontiguousarray(value).tobytes())
                        for read, target, index, score in zip(chunk, truth, prediction, probabilities.max(-1)):
                            writer.writerow((read.read_id, target, index, names.get(int(index), "unassigned"), float(score)))
            finally:
                for handle in handles.values():
                    handle.close()
        result = metrics.result()
        # np.savez writes disk-backed arrays in bounded chunks. Large logits
        # are never concatenated in RAM; --skip_save_logits never stores them.
        views = {key: np.memmap(work / (key + ".bin"), mode="r", dtype=dtype, shape=(metrics.count, *shape))
                 for key, (dtype, shape) in arrays.items()}
        np.savez_compressed(output / f"predictions{suffix}.npz", **views)
        del views
        (work / "read_predictions.tsv").replace(output / f"read_predictions{suffix}.tsv")
        np.save(output / "confusion_matrix.npy", metrics.confusion)
        support, predicted, precision_values, recall, f1 = metrics.per_class()
        with (output / "classification_report.csv").open("w", newline="") as dest:
            writer = csv.writer(dest)
            writer.writerow(("class_id", "name", "precision", "recall", "f1-score", "support"))
            for index in np.flatnonzero(support + predicted):
                writer.writerow((index, names.get(int(index), str(index)), precision_values[index], recall[index], f1[index], support[index]))
        with (output / "top_confusions.csv").open("w", newline="") as dest:
            writer = csv.writer(dest)
            writer.writerow(("true_class_id", "predicted_class_id", "count"))
            matrix = metrics.confusion.copy()
            np.fill_diagonal(matrix, 0)
            for offset in np.argsort(matrix.ravel())[-50:][::-1]:
                i, j = divmod(int(offset), num_classes)
                if matrix[i, j]:
                    writer.writerow((i, j, int(matrix[i, j])))
        elapsed = time.perf_counter() - started
        result.update(checkpoint_epoch=epoch, max_token_length=max_length, task=task, rc_tta=args.rc_tta,
                      precision=precision, aggregation="mean_logits", num_reads=metrics.count,
                      encoder_fastpath=False if precision == "fp32" else torch.backends.mha.get_fastpath_enabled(),
                      pipeline_reads_per_sec=metrics.count / elapsed,
                      tokenize_and_forward_reads_per_sec=metrics.count / max(inference_time, 1e-12),
                      timing_scope="label indexing, input parsing, tokenization, inference and output serialization; model loading excluded",
                      streaming=True)
        (output / f"eval_metrics{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.rc_tta:
            (output / "eval_metrics_fwd_only.json").write_text(json.dumps(forward_metrics.result(), indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return result
