"""Streaming FASTA/FASTQ input, including gzip-compressed files."""
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
from pathlib import Path
import sys


@dataclass(frozen=True)
class Read:
    read_id: str
    sequence: str


@contextmanager
def open_text(path):
    if str(path) == "-":
        yield sys.stdin
        return
    path = Path(path)
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as source:
        yield source


def _read(header, pieces, normalize):
    name = header[1:].strip()
    sequence = "".join(pieces)
    if not name or not sequence:
        raise ValueError("Each read requires a nonempty identifier and sequence")
    if normalize:
        sequence = sequence.upper()
        invalid = set(sequence) - set("ACGTNRYMKSWBDHV")
        if invalid:
            raise ValueError(f"Read {name!r} contains invalid DNA symbols: {sorted(invalid)}")
    return Read(name, sequence)


def iter_reads(path, *, normalize=True):
    """Yield records in input order, validating multiline FASTQ quality lengths.

    Identifiers retain the full header after '>'/'@'. No reads are silently
    dropped; quality scores are validated and are not model inputs.
    """
    with open_text(path) as source:
        first = next((line.rstrip("\r\n") for line in source if line.strip()), None)
        if first is None:
            raise ValueError(f"{path}: input contains no reads")
        if first.startswith(">"):
            header, pieces = first, []
            for raw in source:
                line = raw.strip()
                if line.startswith(">"):
                    yield _read(header, pieces, normalize)
                    header, pieces = line, []
                elif line:
                    pieces.append(line)
            yield _read(header, pieces, normalize)
        elif first.startswith("@"):
            header = first
            while header:
                if not header.startswith("@"):
                    raise ValueError(f"{path}: expected FASTQ '@' header")
                pieces = []
                for raw in source:
                    line = raw.rstrip("\r\n")
                    if line.startswith("+"):
                        break
                    pieces.append(line)
                else:
                    raise ValueError(f"{path}: FASTQ record lacks '+' separator")
                read = _read(header, pieces, normalize)
                quality_length = 0
                while quality_length < len(read.sequence):
                    quality = source.readline()
                    if not quality:
                        raise ValueError(f"{path}: truncated FASTQ quality for {read.read_id!r}")
                    quality_length += len(quality.rstrip("\r\n"))
                if quality_length != len(read.sequence):
                    raise ValueError(f"{path}: sequence/quality length mismatch for {read.read_id!r}")
                yield read
                following = source.readline()
                if following == "":
                    break
                header = following.rstrip("\r\n")
                if not header:
                    raise ValueError(f"{path}: blank line between FASTQ records")
        else:
            raise ValueError(f"{path}: expected FASTA '>' or FASTQ '@' header")


def batches(records, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def reverse_complement(sequence):
    return sequence.upper().translate(str.maketrans("ACGTNRYMKSWBDHV", "TGCANYRKMSWVHDB"))[::-1]
