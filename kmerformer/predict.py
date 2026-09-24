"""Classify unlabelled DNA reads with a verified KmerFormer model bundle."""
import argparse
from contextlib import contextmanager, nullcontext
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import threading

import torch
from .reads import Read, batches, iter_reads, reverse_complement

_INFERENCE_LOCK = threading.RLock()


@contextmanager
def inference_kernels(precision):
    """Use the unfused encoder for FP32 and restore the host setting on exit.

    The fused CUDA encoder in PyTorch 2.6 changed scores by up to 0.0021 on
    the trained example; the unfused path agrees with CPU within 1e-6. This
    affects the new prediction reference, not historical benchmark timings.
    PyTorch exposes this setting globally, so our inference calls serialize
    its changes. Calls made directly to unrelated models remain caller-owned.
    """
    with _INFERENCE_LOCK:
        previous = torch.backends.mha.get_fastpath_enabled()
        matmul_precision = torch.get_float32_matmul_precision()
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            if precision == "fp32":
                torch.backends.mha.set_fastpath_enabled(False)
                torch.set_float32_matmul_precision("highest")
                torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)
            torch.set_float32_matmul_precision(matmul_precision)
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def logits_for_sequences(model, tokenizer, sequences, *, device, max_length,
                         rc_tta=False, precision="fp32", kmer_preprocess=None):
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if device.type != "cuda" and precision == "fp16":
        raise ValueError("fp16 inference requires CUDA; use fp32 on CPU")
    def one_orientation(strings):
        if kmer_preprocess:
            k, stride = kmer_preprocess["k"], kmer_preprocess.get("stride", 1)
            strings = [" ".join(s[i:i+k] for i in range(0, len(s)-k+1, stride)) if len(s) >= k else s for s in strings]
        encoded = tokenizer(strings, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        with inference_kernels(precision), torch.inference_mode(), (nullcontext() if precision == "fp32" else torch.amp.autocast(
                device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)):
            return model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device)).float()
    forward = one_orientation(sequences)
    reverse = one_orientation([reverse_complement(s) for s in sequences]) if rc_tta else None
    logits = (forward + reverse) / 2 if rc_tta else forward
    if not torch.isfinite(logits).all():
        raise ValueError("Inference produced non-finite logits")
    return logits, forward, reverse


def predict_records(bundle, records, *, batch_size=128, rc_tta=None, precision="fp32"):
    rc_tta = bundle.protocol["rc_tta"] if rc_tta is None else rc_tta
    for chunk in batches(records, batch_size):
        if not all(isinstance(read, Read) for read in chunk):
            raise TypeError("predict expects Read records; use iter_reads(path) for FASTA/FASTQ")
        sequences = []
        for read in chunk:
            sequence = read.sequence.upper()
            if not read.read_id or not sequence or set(sequence) - set("ACGTNRYMKSWBDHV"):
                raise ValueError("Prediction requires nonempty read IDs and IUPAC DNA sequences")
            sequences.append(sequence)
        logits, _, _ = logits_for_sequences(bundle.model, bundle.tokenizer, sequences,
            device=bundle.device, max_length=bundle.max_length, rc_tta=rc_tta, precision=precision,
            kmer_preprocess=bundle.config["data"].get("kmer_preprocess"))
        scores, ids = logits.softmax(-1).max(-1)
        for read, index, score in zip(chunk, ids.cpu().tolist(), scores.cpu().tolist()):
            label = bundle.labels[index]
            yield {"read_id": read.read_id, "class_id": index, "taxon_name": label["name"] or "unassigned",
                   "rank": label["rank"], "score": score, "model_id": bundle.manifest["model_id"]}


def main():
    from .bundle import ModelBundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Verified local bundle directory")
    parser.add_argument("--reads", required=True, help="FASTA/FASTQ, optionally gzip compressed; '-' reads stdin")
    parser.add_argument("--output", required=True, help="Output TSV; a JSON protocol record is written alongside it")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda[:index]")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--rc-tta", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")
    output = Path(args.output).resolve()
    metadata_path = Path(str(output) + ".json")
    if (output.exists() or metadata_path.exists()) and not args.overwrite:
        parser.error("Output exists; choose a new path or pass --overwrite")
    if args.reads != "-" and output == Path(args.reads).resolve():
        parser.error("Output cannot replace the input reads")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    bundle = ModelBundle(args.model, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    started, count = time.perf_counter(), 0
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "w", newline="") as dest:
            writer = csv.DictWriter(dest, fieldnames=("read_id", "class_id", "taxon_name", "rank", "score", "model_id"), delimiter="\t")
            writer.writeheader()
            for row in bundle.predict(iter_reads(args.reads), batch_size=args.batch_size,
                                      rc_tta=args.rc_tta, precision=args.precision):
                writer.writerow(row)
                count += 1
        if not count:
            raise ValueError("Input contains no reads")
        elapsed = time.perf_counter() - started
        record = {"model_id": bundle.manifest["model_id"], "read_count": count,
                  "device": str(bundle.device), "precision": args.precision,
                  "encoder_fastpath": False if args.precision == "fp32" else torch.backends.mha.get_fastpath_enabled(),
                  "rc_tta": bundle.protocol["rc_tta"] if args.rc_tta is None else args.rc_tta,
                  "aggregation": "mean_logits", "batch_size": args.batch_size,
                  "elapsed_seconds": elapsed, "pipeline_reads_per_second": count / elapsed,
                  "timing_scope": "read parsing, tokenization, forward/RC inference and TSV writing; excludes model loading",
                  "score_semantics": "closed-catalogue softmax score; not calibrated novelty detection",
                  "class_id_namespace": "training catalogue, not NCBI taxonomy IDs"}
        os.replace(temporary, output)
        metadata_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
