# Configs

One YAML per arm in the paper. Each file opens with a `# RESULT:` banner giving the
number it produced, so a config and its result cannot drift apart; the prose below the
banner is the rationale written when the run was commissioned, kept as provenance.

Paths use `${KF_DATA}`, `${KF_OUT}` and `${KF_VOCAB}`, expanded at load time — see
[`../docs/DATA.md`](../docs/DATA.md). Unset variables are left verbatim, so a missing
one fails as `no such file: $KF_DATA/reads_50M.fa` rather than reading from `/`.

```
6mer/               Part 2 -- shared non-overlapping 6-mer, architecture varies
  L{1,8,16,29}_50M      the depth series; L29 is the headline 6-mer arm
  L16_50M_stride1       stride-1 control: 146 tokens, 5.19 points WORSE
  data_scale/           Part 1 -- the same arms at 500K / 1M / 5M reads
13mer/              Part 3 -- the vocabulary stops being free
  exact_1L_50M_meanpool   the headline 13-mer arm, 91.15%
  exact_1L_250M_meanpool  the same at MetaTransformer's own budget, 99.16%
  exact_1L_50M_attnpool   pooling ablation
  exact_16L_50M           depth hurts at k=13
  hashed_d{64,128}_*      2 GiB table instead of 8; the recommended recipe
baselines/          Part 1 -- NT-v2 at four data budgets, DNABERT-1/-2 at 5M
ablations/          controls: RC, pooling, repeats, conv front-end,
                    and the four steps walking our config toward MetaTransformer
bench/              throughput and peak memory (main-text Table 2)
extra/              secondary or exploratory settings; the two NT-v2 250M runs
                    are reported in Part 1 on a separate archived test pool
```

`extra/` contains secondary configurations. A file's `# STATUS:` line describes
whether an output has been located; the two NT-v2 250M runs are now in the manuscript. Note the
distinction that line now draws: **"no output found on any reachable machine" is not
"never ran"**. Conflating the two is a mistake this repo already made once — the two
`ntv2_lora_250M*` arms were written up as unrun and had in fact completed, on a
cluster no longer reachable from here. Their scores are in
[`../docs/RESULTS.md`](../docs/RESULTS.md#nt-v2-at-the-250m-budget).

## Tested invariants

`tests/test_configs.py` walks this tree and enforces what makes it usable off the
machine the runs were done on, so a future edit cannot quietly break it:

- every file is valid YAML, loads through `load_config`, and declares an output dir;
- **no absolute path is baked in** — every path goes through `${KF_DATA}`,
  `${KF_OUT}` or `${KF_VOCAB}`, and an unrecognised variable is an error;
- output dirs are unique, so two configs cannot overwrite each other's checkpoints
  and have the second look like a reproduction of the first;
- every config outside `bench/` and `extra/` carries a `# RESULT:` banner;
- `sparse_embedding` appears only on exact-vocabulary arms, since that flag is both
  what makes the 33.5M-row table trainable and what rules out data parallelism;
- and separately, `train_ddp.rejects_data_parallel` keys on **that flag and nothing
  else** — checked in both directions against every config here. Rejecting a sparse
  arm prevents an opaque reducer failure; *admitting* the non-sparse ones is what
  keeps multi-GPU training available to the hashed 13-mer arms, which are long-*k*
  too. A guard rewritten to key on *k* would break that silently, so the test is
  mutation-checked rather than assumed;
- any custom k-mer tokenizer declares `model.vocab_size` (without it the embedding
  is sized from NT-v2's 6-mer vocabulary, which is a silently wrong model rather
  than an error);
- every hashed arm uses the same 2^22 buckets, so they stay comparable;
- every config path linked from the docs exists.

## Which config for which claim

[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) maps every claim in the paper to the
config and command that produce it. [`../docs/RESULTS.md`](../docs/RESULTS.md) has the
numbers.

## Anatomy

```yaml
model:
  type: shallow_transformer     # or omit for the LoRA-adapted GFM path
  backbone: InstaDeepAI/...     # tokenizer source at k=6; unused at k=13
  head_type: attention_pool     # or mean_pool -- better at k=13
  max_seq_len: 150
  vocab_size: 33545102          # set only for a custom k-mer tokenizer; asserted
  shallow_config:
    d_model: 64                 # 128 at k=6; 64 at k=13, forced by table size
    num_layers: 1               # free -- this is what Part 2 measures
    sparse_embedding: true      # exact 13-mer only; forces single-GPU training
    conv_kernels: null          # the negative-result front-end; off by default
data:
  tokenizer:                    # omit entirely to use the backbone's own tokenizer
    type: exact_kmer            # or hashed_kmer
    k: 13
    stride: 1                   # stride k gives non-overlapping tokens
  lazy: true                    # stream from a byte-offset index; needed above 5M
  max_token_length: 140         # 138 stride-1 13-mers + [CLS] + one PAD
  rc_augment: true
training:
  amp_dtype: fp16               # kept over bf16 so A6000 runs match V100 ones
  periodic_save_interval: 40000 # so a multi-day run resumes rather than restarts
  num_workers: 4                # set 0 for the large-vocabulary arms; see below
```

Two settings bite in practice.

`num_workers` — for the exact-13-mer arms, set it to **0**. Resident memory is ~18 GiB
per worker because each holds the 256 MiB lookup table plus its share of the pool, and
four workers exhausted 62 GiB of host RAM mid-epoch in the original runs.

`sparse_embedding` — required for the exact-13-mer table, where a dense gradient would be
table-sized, and it forces single-GPU training because torch's data-parallel reducer
rejects sparse gradients. `train_ddp` checks for it and exits with that explanation
rather than failing inside the reducer an hour into a run.

## Adding an arm

Copy the nearest config, change one thing, and give `output.dir` a new name under
`${KF_OUT}`. Keeping one variable moving per file is what made the paper's comparisons
attributable — several of its configs exist only so that a pair of runs differs in
exactly one setting.
