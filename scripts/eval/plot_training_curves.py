#!/usr/bin/env python
"""Plot v14/v15 training curves (val_acc + train_acc per epoch)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CKPT = os.environ.get("KF_OUT", ".")
runs = {
    "v14 (250M, from scratch)":  (f"{CKPT}/nt_token_genus_v14_250M_balanced/training_history.csv", "#1f77b4"),
    "v15 (250M, warm-start v9)": (f"{CKPT}/nt_token_genus_v15_250M_warmstart/training_history.csv", "#d62728"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for name, (path, c) in runs.items():
    df = pd.read_csv(path)
    ax1.plot(df["epoch"], df["val_acc"]*100, "-o", color=c, label=name, ms=3)
    ax2.plot(df["epoch"], df["train_acc"]*100, "-o", color=c, label=name, ms=3)
    # annotate final val_acc
    ax1.annotate(f"{df['val_acc'].iloc[-1]*100:.2f}%",
                 (df["epoch"].iloc[-1], df["val_acc"].iloc[-1]*100),
                 textcoords="offset points", xytext=(5, 0), fontsize=9, color=c)

# v9 reference (trained on TWCC; final balanced-val ~66.29%)
for ax in (ax1,):
    ax.axhline(66.29, ls="--", color="gray", lw=1, alpha=0.7)
    ax.text(1, 66.5, "v9 final (66.29%)", fontsize=8, color="gray")

ax1.set_title("Validation accuracy per epoch")
ax1.set_xlabel("epoch"); ax1.set_ylabel("val_acc (%)"); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)
ax2.set_title("Train accuracy per epoch")
ax2.set_xlabel("epoch"); ax2.set_ylabel("train_acc (%)"); ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

fig.suptitle("NT-v2 Genus training curves — v14 vs v15 (balanced val during training)", fontsize=12)
fig.tight_layout()
out = os.path.join(CKPT, "training_curves_v14_v15.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)

# also print the numeric tails for sanity
for name, (path, _) in runs.items():
    df = pd.read_csv(path)
    print(f"\n{name}: {len(df)} epochs, final val_acc={df['val_acc'].iloc[-1]*100:.2f}%, "
          f"best val_acc={df['val_acc'].max()*100:.2f}% @ep{df['val_acc'].idxmax()+1}")
