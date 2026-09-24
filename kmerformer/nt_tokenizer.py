"""Fixed-vocabulary tokenizer compatible with the released NT-v2 6-mer IDs.

    The vocabulary and its source revision are supplied as JSON. The tokenizer
    performs greedy longest-token matching, including single-base fallbacks;
    it loads no Hugging Face model or remote Python code.
"""
import json
from pathlib import Path
import numpy as np


class NucleotideTokenizer:
    def __init__(self, vocab_path):
        self.vocab_path = str(vocab_path)
        self.spec = json.loads(Path(vocab_path).read_text())
        self.all_tokens = self.spec["tokens"]
        if len(set(self.all_tokens)) != len(self.all_tokens):
            raise ValueError("Tokenizer vocabulary has duplicate tokens")
        self.vocab = {token: i for i, token in enumerate(self.all_tokens)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.trie = {}
        for token, index in self.vocab.items():
            node = self.trie
            for char in token:
                node = node.setdefault(char, {})
            node[None] = index

    def _match(self, text, offset):
        node, found = self.trie, None
        for end in range(offset, len(text)):
            node = node.get(text[end])
            if node is None:
                break
            if None in node:
                found = (node[None], end + 1)
        return found

    def _encode(self, text):
        ids, offset = [], 0
        while offset < len(text):
            if text[offset].isspace():
                offset += 1
                continue
            found = self._match(text, offset)
            if found:
                token, offset = found
                ids.append(token)
            else:
                offset += 1
                while offset < len(text) and not text[offset].isspace() and not self._match(text, offset):
                    offset += 1
                ids.append(self.unk_token_id)
        return ids

    def __call__(self, text, max_length=None, padding=False, truncation=False,
                 return_tensors=None, **kwargs):
        single = isinstance(text, str)
        rows = [[self.cls_token_id] + self._encode(s) for s in ([text] if single else text)]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        length = max_length if padding == "max_length" else max(map(len, rows), default=0)
        masks = [[1] * len(row) for row in rows]
        if padding:
            if length is None:
                raise ValueError("max_length is required for max_length padding")
            for row, mask in zip(rows, masks):
                if len(row) > length:
                    raise ValueError("Sequence exceeds padding length; enable truncation")
                mask.extend([0] * (length - len(row)))
                row.extend([self.pad_token_id] * (length - len(row)))
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long)}
        return {"input_ids": rows[0] if single else rows,
                "attention_mask": masks[0] if single else masks}
