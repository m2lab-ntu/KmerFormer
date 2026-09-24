"""KmerFormer models, fixed tokenizers and portable inference bundles."""

__version__ = "0.2.0rc1"

from .kmer_tokenizers import (  # noqa: F401
    ExactKmerTokenizer,
    HashedKmerTokenizer,
    build_tokenizer,
)
from .model import (  # noqa: F401
    ShallowTransformerClassifier,
    TokenLevelGFMClassifier,
    create_model,
)

__all__ = [
    "ExactKmerTokenizer",
    "HashedKmerTokenizer",
    "build_tokenizer",
    "ShallowTransformerClassifier",
    "TokenLevelGFMClassifier",
    "create_model",
]

from .model import KmerFormer
from .reads import Read, iter_reads

__all__ += ["ModelBundle", "KmerFormer", "Read", "iter_reads"]


def __getattr__(name):
    if name == "ModelBundle":
        from .bundle import ModelBundle
        return ModelBundle
    raise AttributeError(name)
