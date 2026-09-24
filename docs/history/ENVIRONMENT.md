# Historical environment specification

Source snapshot: `5a9870d`. These recorded dependency versions describe the
previous code delivery; they are not the supported installation environment.
In particular, PyTorch 2.5.1 and Transformers 4.46.3 have published security
advisories. Use the current package for inference and new work.

Exact historical execution also requires the corresponding source revision,
input manifests and evaluation precision. Archived settings do not establish
that every training run used an identical environment.

## Conda specification

```yaml
# The environment the reported runs were trained and evaluated in
# (RTX A6000 48 GiB, CUDA 12.4).  Create with:
#
#     conda env create -f environment.yml
#     conda activate kmerformer
#     pip install -e .
#
# transformers is PINNED and the pin is load-bearing: 5.x removed
# `find_pruneable_heads_and_indices`, which NT-v2's vendored `modeling_esm.py`
# imports through trust_remote_code, so every foundation-model baseline dies on
# import. Nothing in this repo needs a newer version.
#
# numpy is held below 2.1 to match the runs. The k-mer tokenizers do fixed-width
# uint64 arithmetic and rely on numpy's wrap-on-overflow behaviour for unsigned
# integers; that is stable across 2.x, but the token ids are the model's input
# vocabulary, so the version that produced the checkpoints is the one recorded.
name: kmerformer
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pytorch=2.5.1
  - pytorch-cuda=12.4
  - pip:
      - transformers==4.46.3
      - peft==0.20.0
      - numpy==2.0.1
      - pandas>=2.2
      - scikit-learn>=1.5
      - scipy>=1.14
      - pyyaml>=6.0
      - tqdm>=4.66
      - matplotlib>=3.9
      - seaborn>=0.13
      - psutil>=5.9
```

## Pip specification

```text
# pip-only install. Assumes a torch matching your CUDA is already present:
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# then:
#   pip install -r requirements.txt && pip install -e .
#
# See environment.yml for why transformers and numpy are pinned.
torch==2.5.1
transformers==4.46.3
peft==0.20.0
numpy==2.0.1
pandas>=2.2
scikit-learn>=1.5
scipy>=1.14
pyyaml>=6.0
tqdm>=4.66
matplotlib>=3.9
seaborn>=0.13
psutil>=5.9
```
