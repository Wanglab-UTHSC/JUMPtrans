from __future__ import annotations

import os, re, json, argparse, math, random, hashlib
import yaml
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

import tensorflow as tf

# ===== FORCE TENSORFLOW CPU ONLY (keep PyTorch on GPU) =====
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    # if TF already initialized or no GPU, ignore
    pass
# ==========================================================

from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, Activation,
                                     Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D,
                                     Add, Concatenate)
from tensorflow.keras.models import Model

# =============== User Config (edit defaults here) ===============
# Path related
DEFAULT_DATASET_CSV = "dataset/new_dataset_with_sequence_all(with_STY).csv"
DEFAULT_FAMILY_MAP_CSV = "dataset/PhosphositePlus_with_family_labeled.csv"
DEFAULT_OUTDIR = "outputs_v23"
DEFAULT_CACHE_DIR = "cache_v16_plus"

# Backbone / input windows
DEFAULT_BACKBONE = "aa_onehot"  # or "aa_onehot"
DEFAULT_WINDOWS = "51,33,15"

# ESM embedding representation
DEFAULT_ESM_REPR_MODE = "pooled"  # "pooled" or "token"
DEFAULT_ESM_CACHE_DTYPE = "float16"  # "float16" or "float32"

# Stage-1 default hyper-parameters
DEFAULT_STAGE1_EPOCHS = 30
DEFAULT_STAGE1_CH = 256
DEFAULT_STAGE1_DROP = 0.3
DEFAULT_STAGE1_LR = 2e-4
DEFAULT_STAGE1_BATCH = 128
DEFAULT_OVERSAMPLER_FAMILY = "smote"  # "none" or "smote"
DEFAULT_USE_CLASS_WEIGHT_FAMILY = True

# Stage-2 default hyper-parameters
DEFAULT_STAGE2_EPOCHS = 20
DEFAULT_STAGE2_CH = 256
DEFAULT_STAGE2_DROP = 0.3
DEFAULT_STAGE2_LR = 2e-4
DEFAULT_STAGE2_BATCH = 128
DEFAULT_OVERSAMPLER_KINASE = "smote"  # "none" or "smote"
DEFAULT_USE_CLASS_WEIGHT_KINASE = True

# Negative sample settings
DEFAULT_NEG_RATIO_MODE = "fixed"  # "fixed"|"linear"|"sqrt"
DEFAULT_NEG_RATIO_FIXED = 0.2
DEFAULT_NEG_RATIO_MINMAX = "0.1,0.5"

# Gating
DEFAULT_TOPK_FAMILY = 3
DEFAULT_SOFT_GATE = True

SEED = 42


def set_global_seed(seed: int):
    """Set global random seed for Python, NumPy and TensorFlow."""
    global SEED
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


# -------------------- small utils --------------------
def safe_dirname(name: str) -> str:
    name = re.sub(r"[:*?<>|]", "_", str(name))
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", "_", name).strip("_")
    return name or "Unknown"


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
AA_TO_ID = {a: i for i, a in enumerate(AA_LIST)}


def aa_onehot(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), len(AA_LIST)), dtype=np.float32)
    for i, ch in enumerate(seq):
        arr[i, AA_TO_ID.get(ch, AA_TO_ID["X"])] = 1.0
    return arr


SITE_RE = re.compile(r"^\s*([STY])\s*-?(\d+)\s*$", re.IGNORECASE)
SPLIT = re.compile(r"[;,|]")

# =============== Simple Run Logger ===============
# Collect key training/eval messages and save to outputs/run_log.txt
from datetime import datetime

RUN_LOG: list[str] = []


def log(msg: str):
    """Append a timestamped message to global RUN_LOG and print to console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    RUN_LOG.append(line)
    print(line)


def keep_first_site(s: str) -> str:
    return re.split(r"[;, ]+", str(s).strip())[0]


def site_ok(s: str):
    m = SITE_RE.match(str(s))
    if not m: return False, "", -1
    aa = m.group(1).upper();
    pos = int(m.group(2))
    return True, aa, pos


def find_col(df: pd.DataFrame, keys: List[str]) -> str:
    for k in keys:
        for c in df.columns:
            if k.lower() == c.lower() or k.lower() in c.lower():
                return c
    raise KeyError(keys)


def normalize_family(raw: str) -> str:
    if raw is None: return "Unknown"
    s_raw = str(raw).strip()
    s = s_raw.lower().replace(" ", "").replace("_", "").replace("-", "")
    alias = {
        "agc": "AGC Ser/Thr", "agcserthr": "AGC Ser/Thr",
        "camk": "CAMK Ser/Thr", "camkserthr": "CAMK Ser/Thr",
        "ck1": "CK1 Ser/Thr", "ck1serthr": "CK1 Ser/Thr",
        "cmgc": "CMGC Ser/Thr", "cmgcserthr": "CMGC Ser/Thr",
        "ste": "STE Ser/Thr", "steserthr": "STE Ser/Thr",
        "tkl": "TKL Ser/Thr", "tklserthr": "TKL Ser/Thr",
        "tyr": "Tyr", "ptk": "Tyr",
        "others": "Others",
        "atypicalalphatype": "Atypical: Alpha-type",
        "atypicalpi3pi4kinase": "Atypical: PI3/PI4-kinase",
    }
    if s in alias: return alias[s]
    if s_raw.lower().startswith("atypical"):
        if "alpha" in s: return "Atypical: Alpha-type"
        if "pi3" in s or "pi4" in s: return "Atypical: PI3/PI4-kinase"
        return "Unknown"
    return s_raw


def load_family_mapping(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if not {"kinase", "kinase-family"}.issubset(df.columns):
        raise ValueError("mapping       : kinase, kinase-family")
    df["kinase"] = df["kinase"].astype(str).str.strip().str.upper()
    df["kinase-family"] = df["kinase-family"].astype(str).str.strip()
    return dict(zip(df["kinase"], df["kinase-family"]))


# -------------------- ESM singleton & caching --------------------
_ESM = {"name": None, "model": None, "batch_converter": None, "alphabet": None}


def get_esm_model(backbone_name: str):
    """Load an ESM model from esm.pretrained by name and cache it."""
    try:
        import torch, esm
    except Exception as e:
        raise ImportError("You must install torch and fair-esm to use ESM backbones.") from e
    backbone_name = str(backbone_name)
    if _ESM["name"] == backbone_name and _ESM["model"] is not None:
        return _ESM["model"], _ESM["batch_converter"], _ESM["alphabet"]
    fn = getattr(esm.pretrained, backbone_name, None)
    if fn is None:
        raise ValueError(f"Unknown ESM backbone name: {backbone_name}. It must be a function in esm.pretrained.")
    model, alphabet = fn()
    model.eval()
    _ESM["name"] = backbone_name
    _ESM["model"] = model
    _ESM["alphabet"] = alphabet
    _ESM["batch_converter"] = alphabet.get_batch_converter()
    return _ESM["model"], _ESM["batch_converter"], _ESM["alphabet"]
    try:
        import torch, esm
    except Exception as e:
        raise ImportError("     torch   fair-esm      --backbone esm2_t33") from e
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    _ESM.update(
        {"loaded": True, "model": model, "batch_converter": alphabet.get_batch_converter(), "alphabet": alphabet})
    return _ESM["model"], _ESM["batch_converter"], _ESM["alphabet"]


def cut_window(seq: str, pos: int, L: int) -> str:
    i = pos - 1;
    half = L // 2
    start = max(0, i - half);
    end = min(len(seq), i + half + 1)
    frag = seq[start:end]
    if len(frag) < L:
        pad_left = max(0, half - i)
        pad_right = L - len(frag) - pad_left
        frag = ("X" * pad_left) + frag + ("X" * pad_right)
    return frag[:L]


def embed_aa(df: pd.DataFrame, windows: List[int], out_bad_center: Path | None) -> Dict[int, np.ndarray]:
    embeds = {}
    for L in windows:
        frags = [cut_window(s, p, L) for s, p in zip(df["SEQUENCE"].tolist(), df["POS"].tolist())]
        bad_center = [i for i, f in enumerate(frags) if f[L // 2] not in ("S", "T", "Y")]
        if out_bad_center and len(bad_center) > 0:
            df.iloc[bad_center].to_csv(out_bad_center, index=False)
        embeds[L] = np.stack([aa_onehot(f) for f in frags], axis=0)
    return embeds


def embed_esm(df: pd.DataFrame, windows: List[int], cache_dir: Path, backbone_name: str,
              repr_mode: str = "pooled", cache_dtype: str = "float16") -> Dict[int, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, batch_converter, alphabet = get_esm_model(backbone_name)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeds = {}
    ids = df["ID"].astype(str).tolist()
    for L in windows:
        mode = str(repr_mode).lower().strip()
        dtype_tag = str(cache_dtype).lower().strip()
        key = hashlib.sha1(
            (",".join(ids) + f"|L={L}|{backbone_name}|mode={mode}|dtype={dtype_tag}").encode()).hexdigest()
        fp = cache_dir / f"{backbone_name}_L{L}_{mode}_{dtype_tag}_{key}.npy"
        if fp.exists():
            arr = np.load(fp)
            # Keep cached dtype (often float16) to save RAM; TF will cast as needed.
            embeds[L] = arr
            continue
        frags = [cut_window(s, p, L) for s, p in zip(df["SEQUENCE"].tolist(), df["POS"].tolist())]
        reps = []
        B = 16
        # with torch.no_grad():
        #     for i in range(0, len(frags), B):
        #         batch = [("id", f) for f in frags[i:i + B]]
        #         _, _, toks = batch_converter(batch)
        #         toks = toks.to(device)
        #         out = model(toks, repr_layers=[33], return_contacts=False)
        #         rep = out["representations"][33][:, 1:-1, :].mean(dim=1).cpu().numpy()
        #         reps.append(rep)
        last_layer = model.num_layers

        with torch.no_grad():
            for i in range(0, len(frags), B):
                batch = [("id", f) for f in frags[i:i + B]]
                _, _, toks = batch_converter(batch)
                toks = toks.to(device)

                # 修改点：使用变量 last_layer
                out = model(toks, repr_layers=[last_layer], return_contacts=False)
                rep_tok = out["representations"][last_layer]
                # rep_tok: (B, L+2, D) including BOS/EOS
                if str(repr_mode).lower() == "token":
                    rep = rep_tok[:, 1:-1, :].cpu().numpy()
                else:
                    rep = rep_tok[:, 1:-1, :].mean(dim=1).cpu().numpy()  # pooled
                reps.append(rep)
        X_raw = np.concatenate(reps, axis=0)
        if str(repr_mode).lower() == "token":
            X = X_raw.astype(np.float32)
        else:
            X = X_raw[:, None, :].astype(np.float32)
        if str(cache_dtype).lower() == "float16":
            X_save = X.astype(np.float16)
        else:
            X_save = X.astype(np.float32)
        np.save(fp, X_save)
        X = X_save
        embeds[L] = X
    return embeds


# -------------------- Data cleaning --------------------
def clean_align(df_raw: pd.DataFrame, fam_map: Dict[str, str], outdir: Path) -> pd.DataFrame:
    prot = find_col(df_raw, ["SUB_ACC_ID", "ACC", "PROTEIN", "SUBSTRA"])
    site = find_col(df_raw, ["SUB_MOD_RSD", "SITE", "MOD_RSD"])
    seq = find_col(df_raw, ["SEQUENCE", "SEQ"])
    kin = find_col(df_raw, ["KIN_GENE", "KINASE"])
    df = df_raw.rename(columns={prot: "SUB_ACC_ID", site: "SUB_MOD_RSD", seq: "SEQUENCE", kin: "KIN_GENE"})
    df = df.dropna(subset=["SUB_ACC_ID", "SUB_MOD_RSD", "SEQUENCE", "KIN_GENE"]).copy()
    df["SUB_ACC_ID"] = df["SUB_ACC_ID"].astype(str).str.strip()
    df["SUB_MOD_RSD"] = df["SUB_MOD_RSD"].astype(str).str.strip()
    df["SEQUENCE"] = df["SEQUENCE"].astype(str).str.upper()
    df["KIN_GENE"] = df["KIN_GENE"].astype(str).str.strip().str.upper()

    #        +     STY
    df["SUB_MOD_RSD"] = df["SUB_MOD_RSD"].map(keep_first_site)
    ok_list = [];
    aa_list = [];
    pos_list = []
    for s in df["SUB_MOD_RSD"].tolist():
        ok, aa, pos = site_ok(s)
        ok_list.append(ok);
        aa_list.append(aa);
        pos_list.append(pos)
    df["ok"] = ok_list;
    df["AA"] = aa_list;
    df["POS"] = pos_list
    bad = df[~df["ok"]].copy()
    if len(bad) > 0:
        (outdir / "bad_sites.csv").parent.mkdir(parents=True, exist_ok=True)
        bad.to_csv(outdir / "bad_sites.csv", index=False)
    df = df[df["ok"]].copy()

    # family
    fam_sets = []
    for kins in df["KIN_GENE"]:
        fams = set()
        items = SPLIT.split(kins) if SPLIT.search(kins) else [kins]
        for k in items:
            fam = normalize_family(fam_map.get(k.strip().upper(), "Unknown"))
            fams.add(fam)
        fam_sets.append(fams)
    df["FAMILY_SET"] = fam_sets
    #    family
    df = df[df["FAMILY_SET"].map(len) == 1].copy()
    df["Family"] = df["FAMILY_SET"].apply(lambda s: list(s)[0])

    #    Unknown/       stratify
    vc = df["Family"].value_counts()
    keep = vc[vc >= 2].index.tolist()
    df = df[df["Family"].isin(keep)].copy()

    # ID
    df["ID"] = df["SUB_ACC_ID"] + "_" + df["SUB_MOD_RSD"]
    df = df.drop_duplicates(subset=["ID", "Family"]).copy()

    return df


# -------------------- Network definition (aligned with v16, detail enhanced) --------------------
def conv_blk(x, ch, k=3, drop=0.2):
    x = Conv1D(ch, k, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.gelu)(x)
    return Dropout(drop)(x)


def se_block(x, r=8):
    c = x.shape[-1]
    s = GlobalAveragePooling1D()(x)
    s = Dense(max(int(c) // r, 4), activation="relu")(s)
    s = Dense(int(c), activation="sigmoid")(s)
    return x * tf.expand_dims(s, 1)


def tower(x, ch=256, drop=0.3):
    y = conv_blk(x, ch, 3, drop)
    y = conv_blk(y, ch, 5, drop)
    res = y
    y = conv_blk(y, ch, 3, drop)
    y = Add()([y, res])
    # Squeeze-Excitation
    y = se_block(y)
    return Concatenate()([GlobalAveragePooling1D()(y), GlobalMaxPooling1D()(y)])


def build_multibranch(n_classes: int, token_dims: List[int], windows: List[int], ch=256, drop=0.3) -> Model:
    inputs = [];
    feats = []
    for L, td in zip(windows, token_dims):
        inp = Input((None, td))  # allow variable time dimension
        f = tower(inp, ch, drop)
        inputs.append(inp);
        feats.append(f)
    m = Concatenate()(feats) if len(feats) > 1 else feats[0]
    x = Dense(512, activation="relu")(m)
    x = Dropout(drop)(x)
    o = Dense(n_classes, dtype="float32")(x)
    return Model(inputs, o)


def build_multibranch_mlp(n_classes: int, token_dims: List[int], windows: List[int], ch=256, drop=0.3) -> Model:
    inputs = []
    feats = []
    for L, td in zip(windows, token_dims):
        inp = Input((None, td))
        x = GlobalAveragePooling1D()(inp)
        feats.append(x)
        inputs.append(inp)

    m = Concatenate()(feats) if len(feats) > 1 else feats[0]
    x = Dense(512, activation="relu")(m)
    x = Dropout(drop)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(drop)(x)
    o = Dense(n_classes, dtype="float32")(x)
    return Model(inputs, o)


def ce(label_smoothing=0.05):
    try:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing)
    except TypeError:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# -------------------- Simple SMOTE and class weight helpers --------------------
def smote_interpolate(Xs: List[np.ndarray], y: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    vals, cnts = np.unique(y, return_counts=True)
    maj = int(cnts.max())
    Xnew = [[X] for X in Xs];
    ylist = [y]
    for c, n in zip(vals, cnts):
        n = int(n)
        if n >= maj: continue
        need = maj - n
        idx = np.where(y == c)[0];
        m = len(idx)
        for _ in range(need):
            a, b = np.random.choice(idx, 2, replace=(m < 2));
            lam = np.random.rand()
            for wi in range(len(Xs)):
                x = lam * Xs[wi][a] + (1 - lam) * Xs[wi][b]
                Xnew[wi].append(x[None, ...])
        ylist.append(np.full((need,), int(c), dtype=np.int32))
    return [np.concatenate(t, 0) for t in Xnew], np.concatenate(ylist, 0)


def build_class_weight(y: np.ndarray, num_classes: int | None = None):
    """
    Build class_weight dict for keras, ensuring keys cover [0..num_classes-1].
    For classes absent in y (e.g. only present in val/test), we set weight=1.0.
    """
    vals, cnts = np.unique(y, return_counts=True)
    total = len(y)
    if len(vals) == 0:
        return {}
    K_obs = len(vals)
    base = {int(v): float(total / (K_obs * c)) for v, c in zip(vals, cnts)}
    if num_classes is None:
        num_classes = int(max(vals)) + 1
    for k in range(num_classes):
        if k not in base:
            base[k] = 1.0
    return base


# -------------------- Evaluation (keep v16, plus additions) --------------------
def plot_confusion(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(classes)), max(7, 0.45 * len(classes))))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True', xlabel='Pred', title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout();
    fig.savefig(save_path, format="pdf");
    plt.close(fig)


def plot_confusion_with_diag(cm, classes, save_path, title: str | None = None):
    """Plot confusion matrix with a highlighted diagonal (no per-cell text)."""
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(classes)), max(7, 0.45 * len(classes))))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True',
        xlabel='Pred',
        title=title if title is not None else "Confusion Matrix (diag highlighted)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # overlay diagonal
    n = cm.shape[0]
    ax.plot(np.arange(n), np.arange(n), linestyle="--", linewidth=0.8, color="red", alpha=0.8)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf")
    plt.close(fig)


def compute_confusion_counts(cm: np.ndarray, class_names: list[str]) -> dict:
    """Compute global and per-class TP/TN/FP/FN from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum())
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn

    global_counts = {
        "TP": int(tp.sum()),
        "FP": int(fp.sum()),
        "FN": int(fn.sum()),
        "TN": int(tn.sum()),
        "total": int(total),
    }

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[str(name)] = {
            "TP": int(tp[i]),
            "FP": int(fp[i]),
            "FN": int(fn[i]),
            "TN": int(tn[i]),
            "support": int(cm[i].sum()),
        }

    return {"global_counts": global_counts, "per_class_counts": per_class}


def plot_pr_roc_micro(Y_true: np.ndarray, Y_prob: np.ndarray, out_dir: Path, prefix: str):
    """
      micro-averaged PR / ROC   :
      - Y_true: (N, C) one-hot
      - Y_prob: (N, C)
             0    1
    """
    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    #           micro-averaging
    y_true_flat = Y_true.ravel()
    y_prob_flat = Y_prob.ravel()
    #
    if y_true_flat.max() == 0 or y_true_flat.min() == 1:
        return

    try:
        prec, rec, _ = precision_recall_curve(y_true_flat, y_prob_flat)
        fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    except Exception:
        return

    try:
        auprc = auc(rec, prec)
        auc_roc = auc(fpr, tpr)
    except Exception:
        auprc = float("nan")
        auc_roc = float("nan")

    # PR
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{prefix} micro-averaged PR (AUPRC={auprc:.3f})")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_pr_micro.pdf")
        plt.close(fig)
    except Exception:
        pass

    # ROC
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC={auc_roc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{prefix} micro-averaged ROC")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_roc_micro.pdf")
        plt.close(fig)
    except Exception:
        pass


def eval_multiclass_block(y_true, logits, classes, out_dir: Path, prefix="family"):
    """
    Generic multiclass evaluation block used for:
      - Stage-1 family classifier (val/test)
      - Per-family kinase experts (val)
    Exports:
      - accuracy, macro/weighted precision/recall/F1
      - classification_report (json)
      - confusion matrix (csv + pdf)
      - confusion matrix with highlighted diagonal (pdf)
      - macro/weighted ROC-AUC & AP (OvR)
      - micro-averaged PR/ROC curves (global)
      - TP/TN/FP/FN (global + per-class) as JSON
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(logits, axis=1)

    # Basic metrics
    acc = (y_pred == y_true).mean()
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Classification report
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    with open(out_dir / f"{prefix}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(out_dir / f"{prefix}_cm.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, classes, out_dir / f"{prefix}_cm.pdf")
    # Diagonal-highlighted view
    plot_confusion_with_diag(cm, classes, out_dir / f"{prefix}_cm_diag.pdf",
                             title="Confusion Matrix (diag highlighted)")

    # TP/TN/FP/FN counts
    counts = compute_confusion_counts(cm, classes)
    with open(out_dir / f"{prefix}_per_class_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

    # OvR ROC/AUC & AP
    Y_true = np.eye(len(classes), dtype=np.int32)[y_true]
    Y_prob = tf.nn.softmax(logits, axis=1).numpy()

    try:
        auc_macro = roc_auc_score(Y_true, Y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(Y_true, Y_prob, average="weighted", multi_class="ovr")
    except Exception:
        auc_macro = float("nan")
        auc_weighted = float("nan")

    try:
        ap_macro = average_precision_score(Y_true, Y_prob, average="macro")
        ap_weighted = average_precision_score(Y_true, Y_prob, average="weighted")
    except Exception:
        ap_macro = float("nan")
        ap_weighted = float("nan")

    metrics = {
        "acc": float(acc),
        "macro": {
            "precision": float(prec_m),
            "recall": float(rec_m),
            "f1": float(f1_m),
        },
        "weighted": {
            "precision": float(prec_w),
            "recall": float(rec_w),
            "f1": float(f1_w),
        },
        "roc_auc_macro": float(auc_macro),
        "roc_auc_weighted": float(auc_weighted),
        "ap_macro": float(ap_macro),
        "ap_weighted": float(ap_weighted),
        "global_counts": counts.get("global_counts", {}),
    }
    with open(out_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Micro-averaged PR/ROC
    plot_pr_roc_micro(Y_true, Y_prob, out_dir, prefix)
    # <--- 新增开始：保存 PR 曲线数据，用于 Auto-F1 阈值 --->
    from sklearn.metrics import precision_recall_curve
    pr_rows = []
    # 针对每个类别计算 P/R/Threshold
    for i, class_name in enumerate(classes):
        # 制作二分类标签：当前类为 1，其他为 0
        y_true_bin = (y_true == i).astype(int)
        y_score_bin = Y_prob[:, i]

        if y_true_bin.sum() == 0: continue  # 防止除以零

        precision, recall, thresholds = precision_recall_curve(y_true_bin, y_score_bin)
        # thresholds 长度比 p/r 少 1，补一个 1.0 对齐
        thresh_list = list(thresholds) + [1.0]

        for p, r, t in zip(precision, recall, thresh_list):
            pr_rows.append({
                "label": class_name,
                "precision": float(p),
                "recall": float(r),
                "threshold": float(t)
            })

    if pr_rows:
        pd.DataFrame(pr_rows).to_csv(out_dir / f"{prefix}_pr_curves.csv", index=False)
        print(f"[Eval] Saved PR curves to {out_dir / f'{prefix}_pr_curves.csv'}")
    # <--- 新增结束 --->

    return {"acc": acc, "macro_f1": f1_m, "weighted_f1": f1_w}


def eval_global_kinase(y_true_idx, y_prob, classes, out_dir: Path, chunk_size: int = 40):
    """
    Global kinase evaluation:
      - accuracy, macro/weighted precision/recall/F1
      - confusion matrix (csv + pdf)
      - confusion matrix with highlighted diagonal
      - macro/weighted ROC-AUC and AP (OvR)
      - per-class PR/ROC curves exported as CSV
      - TP/TN/FP/FN (global + per-class)
      - large confusion matrix additionally split into smaller heatmaps (chunks)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(y_prob, axis=1)

    acc = (y_pred == y_true_idx).mean()
    prec_m = precision_score(y_true_idx, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true_idx, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true_idx, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_true_idx, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true_idx, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true_idx, y_pred, average="weighted", zero_division=0)

    labels = list(range(len(classes)))
    report = classification_report(
        y_true_idx,
        y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    with open(out_dir / "kinase_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Full confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred, labels=labels)
    np.savetxt(out_dir / "kinase_cm.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, classes, out_dir / "kinase_cm.pdf")
    plot_confusion_with_diag(cm, classes, out_dir / "kinase_cm_diag.pdf",
                             title="Global Kinase Confusion (diag highlighted)")

    # TP/TN/FP/FN counts
    counts = compute_confusion_counts(cm, classes)
    with open(out_dir / "kinase_per_class_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

    # Split large matrix into chunks for easier visualization
    try:
        n_classes = len(classes)
        chunk_size = int(chunk_size) if chunk_size and chunk_size > 0 else 40
        chunk_dir = out_dir / "heatmap_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, n_classes, chunk_size):
            end = min(start + chunk_size, n_classes)
            sub_cm = cm[start:end, start:end]
            sub_classes = classes[start:end]
            tag = f"{start:03d}_{end - 1:03d}"
            np.savetxt(chunk_dir / f"kinase_cm_{tag}.csv", sub_cm, fmt="%d", delimiter=",")
            plot_confusion(sub_cm, sub_classes, chunk_dir / f"kinase_cm_{tag}.pdf")
    except Exception as e:
        print(f"[WARN] Failed to generate chunked kinase heatmaps: {e}")

    # OvR ROC/AUC + AP
    Y_true = np.eye(len(classes), dtype=np.int32)[y_true_idx]
    Y_prob = y_prob.astype(np.float32)

    try:
        auc_macro = roc_auc_score(Y_true, Y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(Y_true, Y_prob, average="weighted", multi_class="ovr")
    except Exception:
        auc_macro = float("nan")
        auc_weighted = float("nan")

    try:
        ap_macro = average_precision_score(Y_true, Y_prob, average="macro")
        ap_weighted = average_precision_score(Y_true, Y_prob, average="weighted")
    except Exception:
        ap_macro = float("nan")
        ap_weighted = float("nan")

    # per-class PR / ROC curves
    from sklearn.metrics import precision_recall_curve, roc_curve
    pr_rows = []
    roc_rows = []

    for idx, name in enumerate(classes):
        y_true_bin = Y_true[:, idx]
        y_score = Y_prob[:, idx]
        pos_cnt = int(y_true_bin.sum())

        if pos_cnt == 0:
            continue
        all_pos = (pos_cnt == len(y_true_bin))

        # PR curve
        try:
            p, r, thr_pr = precision_recall_curve(y_true_bin, y_score)
            thr_list = list(thr_pr) + [1.0]
            for pi, ri, ti in zip(p, r, thr_list):
                pr_rows.append({
                    "label": name,
                    "precision": float(pi),
                    "recall": float(ri),
                    "threshold": float(ti),
                })
        except Exception:
            pass

        # ROC curve
        if all_pos:
            continue
        try:
            fpr, tpr, thr_roc = roc_curve(y_true_bin, y_score)
            for fi, ti, th in zip(fpr, tpr, thr_roc):
                roc_rows.append({
                    "label": name,
                    "fpr": float(fi),
                    "tpr": float(ti),
                    "threshold": float(th),
                })
        except Exception:
            pass

    if pr_rows:
        pd.DataFrame(pr_rows).to_csv(out_dir / "kinase_pr_curves.csv", index=False)
    if roc_rows:
        pd.DataFrame(roc_rows).to_csv(out_dir / "kinase_roc_curves.csv", index=False)

    metrics = {
        "acc": float(acc),
        "macro": {"precision": float(prec_m), "recall": float(rec_m), "f1": float(f1_m)},
        "weighted": {"precision": float(prec_w), "recall": float(rec_w), "f1": float(f1_w)},
        "roc_auc_macro": float(auc_macro),
        "roc_auc_weighted": float(auc_weighted),
        "ap_macro": float(ap_macro),
        "ap_weighted": float(ap_weighted),
        "global_counts": counts.get("global_counts", {}),
    }
    with open(out_dir / "kinase_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Micro-averaged PR/ROC curves
    plot_pr_roc_micro(Y_true, Y_prob, out_dir, prefix="kinase_global")
    return {"acc": acc, "macro_f1": f1_m, "weighted_f1": f1_w}


def eval_per_family_kinase_heatmap(
        df_te_fam: pd.DataFrame,
        probs: np.ndarray,
        kin_names: List[str],
        out_dir: Path,
        fam_name: str,
):
    """
    Per-family kinase evaluation on test set:
      - derive y_true from KIN_GENE that map uniquely into kin_names
      - per-family confusion matrix (csv + pdf)
      - diagonal-highlighted confusion matrix
      - per-family classification report + metrics
      - per-family TP/TN/FP/FN (global + per-kinase)
      - per-family micro-averaged PR / ROC curves
    """
    y_true = []
    rows = []
    idx_map = {k: i for i, k in enumerate(kin_names)}

    # From test subset, find samples that can be uniquely mapped to kin_names using KIN_GENE
    genes_list = df_te_fam["KIN_GENE"].astype(str).tolist()
    for i, genes in enumerate(genes_list):
        if SPLIT.search(genes):
            items = SPLIT.split(genes)
        else:
            items = [genes]
        pick = None
        for g in items:
            g = g.strip().upper()
            if g in idx_map:
                pick = g
                break
        if pick is not None:
            y_true.append(idx_map[pick])
            rows.append(i)

    # If too few samples, skip evaluation and summary for this family
    if len(rows) < 2:
        return None

    y_true = np.array(y_true, dtype=np.int32)
    y_prob = probs[rows]
    y_pred = np.argmax(y_prob, axis=1)

    labels = list(range(len(kin_names)))
    fam_safe = safe_dirname(fam_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(out_dir / f"{fam_safe}_kinase_cm.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, kin_names, out_dir / f"{fam_safe}_kinase_cm.pdf")
    plot_confusion_with_diag(cm, kin_names, out_dir / f"{fam_safe}_kinase_cm_diag.pdf",
                             title=f"{fam_name} kinase confusion (diag highlighted)")

    # Metrics + classification report
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=kin_names,
        zero_division=0,
        output_dict=True,
    )
    with open(out_dir / f"{fam_safe}_kinase_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # TP/TN/FP/FN for this family
    counts = compute_confusion_counts(cm, kin_names)
    with open(out_dir / f"{fam_safe}_kinase_per_class_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

    metrics = {
        "family": str(fam_name),
        "n_samples": int(len(y_true)),
        "n_kinases": int(len(kin_names)),
        "macro_precision": float(prec_m),
        "macro_recall": float(rec_m),
        "macro_f1": float(f1_m),
        "weighted_precision": float(prec_w),
        "weighted_recall": float(rec_w),
        "weighted_f1": float(f1_w),
        "global_counts": counts.get("global_counts", {}),
    }
    with open(out_dir / f"{fam_safe}_kinase_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Per-family micro-averaged PR / ROC curves
    Y_true = np.eye(len(kin_names), dtype=np.int32)[y_true]
    Y_prob = y_prob.astype(np.float32)
    try:
        plot_pr_roc_micro(Y_true, Y_prob, out_dir, prefix=f"{fam_safe}_kinase")
    except Exception as e:
        print(f"[WARN] plot_pr_roc_micro failed for family {fam_name}: {e}")

    return metrics


def train_family(df_tr: pd.DataFrame, windows: List[int], backbone: str, cache_dir: Path,
                 esm_repr_mode: str = "pooled", esm_cache_dtype: str = "float16",
                 use_smote=True, use_class_weight=True, ch=256, drop=0.3, lr=2e-4, epochs=30, batch_size=128,
                 out_dir: Path = Path("outputs_v16_plus/stage1_family")):
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[Stage-1] Start family training: N={len(df_tr)}, backbone={backbone}, windows={windows}, epochs={epochs}, batch_size={batch_size}, use_smote={use_smote}, use_class_weight={use_class_weight}")
    # Embedding
    if backbone.lower() == "aa_onehot":
        embeds = embed_aa(df_tr, windows, out_dir / "bad_sites_center.csv")
    else:
        embeds = embed_esm(df_tr, windows, cache_dir, backbone, repr_mode=esm_repr_mode, cache_dtype=esm_cache_dtype)
    token_dims = [embeds[L].shape[-1] for L in windows]
    # For ESM, time dimension comes from embedding (often 1), avoiding mismatch with window length
    if backbone.lower() == "aa_onehot":
        model_windows = windows
    else:
        model_windows = [embeds[L].shape[1] for L in windows]
    Xs = [embeds[L] for L in windows]

    log(f"[Stage-1] X branch shapes: {[x.shape for x in Xs]}")
    log(f"[Stage-1] time_dims: {[x.shape[1] for x in Xs]}")

    le = LabelEncoder();
    y = le.fit_transform(df_tr["Family"].tolist());
    classes = le.classes_.tolist()

    # Train/val split
    idx = np.arange(len(y))
    strat = y if all(np.bincount(y) > 1) else None
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strat)
    Xtr = [Xi[tr_idx] for Xi in Xs];
    Xva = [Xi[va_idx] for Xi in Xs]
    ytr, yva = y[tr_idx], y[va_idx]

    if use_smote:
        # Old-style SMOTE: flatten all branches and oversample jointly in feature space
        n_samples = Xtr[0].shape[0]
        flat_list = []
        for Xi in Xtr:
            flat_list.append(Xi.reshape(n_samples, -1))
        X_flat = np.concatenate(flat_list, axis=1)
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=SEED)
        X_flat_res, y_res = sm.fit_resample(X_flat, ytr)
        # Reshape back to multi-branch tensors
        Xtr_res = []
        start = 0
        for Xi in Xtr:
            L = Xi.shape[1];
            D = Xi.shape[2]
            size = L * D
            sub = X_flat_res[:, start:start + size]
            Xtr_res.append(sub.reshape(-1, L, D))
            start += size
        Xtr, ytr = Xtr_res, y_res
    class_weight = build_class_weight(ytr, num_classes=len(classes)) if use_class_weight else None

    time_dims = [Xi.shape[1] for Xi in Xs]
    use_mlp = all(t == 1 for t in time_dims)

    if use_mlp:
        log("[Stage-1] Using MLP head (pooled embeddings detected).")
        model = build_multibranch_mlp(len(classes), token_dims, model_windows, ch, drop)
    else:
        log("[Stage-1] Using Conv head (sequence embeddings detected).")
        model = build_multibranch(len(classes), token_dims, model_windows, ch, drop)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=ce(0.05), metrics=["accuracy"])

    # Manual training loop to avoid GPU/CPU copy issues in some environments
    best = (-1.0, None)
    n_tr = Xtr[0].shape[0]
    idx_all = np.arange(n_tr)

    for ep in range(1, epochs + 1):
        # Shuffle indices once per epoch
        np.random.shuffle(idx_all)
        n_batches = int(math.ceil(n_tr / batch_size))
        for bi in range(n_batches):
            b_idx = idx_all[bi * batch_size:(bi + 1) * batch_size]
            if len(b_idx) == 0:
                continue
            batch_X = [Xi[b_idx] for Xi in Xtr]
            batch_y = ytr[b_idx]
            if use_class_weight and class_weight is not None and len(class_weight) > 0:
                sw = np.array([class_weight[int(c)] for c in batch_y], dtype="float32")
            else:
                sw = None
            #
            if sw is not None:
                model.train_on_batch(batch_X, batch_y, sample_weight=sw)
            else:
                model.train_on_batch(batch_X, batch_y)

        # After each epoch, evaluate weighted F1 on validation set
        logits = model.predict(Xva, batch_size=256, verbose=0)
        f1_w = f1_score(yva, np.argmax(logits, 1), average="weighted", zero_division=0)
        log(f"[Stage-1] Epoch {ep}/{epochs}, val_weighted_f1={f1_w:.4f}, best={best[0]:.4f}")
        if f1_w > best[0]:
            best = (f1_w, model.get_weights())
    if best[1] is not None:
        model.set_weights(best[1])

    # Export validation evaluation (same as v16)
    eval_multiclass_block(yva, model.predict(Xva, 256, 0), classes, out_dir / "val", prefix="family")

    # Full-train logits (for gating reference)
    logits_tr = model.predict(Xs, batch_size=256, verbose=0)
    return model, le, classes, (Xs, y, df_tr, logits_tr)


def neg_ratio_value(mode: str, fixed: float, n_pos: int, minmax=(0.1, 0.5)) -> float:
    lo, hi = minmax
    if mode == "fixed":
        return max(0.0, float(fixed))
    elif mode == "linear":
        r = min(hi, max(lo, n_pos / 1000.0))
        return float(r)
    elif mode == "sqrt":
        r = min(hi, max(lo, math.sqrt(max(1, n_pos)) / 50.0))
        return float(r)
    return max(0.0, float(fixed))


def sample_negatives(df_all: pd.DataFrame, fam: str, n_pos: int, ratio: float) -> pd.DataFrame:
    pool = df_all[df_all["Family"] != fam]
    n = max(1, int(round(n_pos * ratio)))
    if len(pool) <= n: return pool.copy()
    return pool.sample(n=n, random_state=SEED)


def train_kinase_expert(df_all: pd.DataFrame, fam: str, windows: List[int], backbone: str, cache_dir: Path,
                        neg_ratio=0.2, use_smote=True, use_class_weight=True, ch=256, drop=0.3,
                        lr=2e-4, epochs=20, batch_size=128, out_dir: Path = Path("outputs_v16_plus/stage2_kinase"),
                        esm_repr_mode: str = "pooled",
                        esm_cache_dtype: str = "float16"
                        ):
    fam_dir = out_dir / safe_dirname(fam);
    fam_dir.mkdir(parents=True, exist_ok=True)
    df_pos = df_all[df_all["Family"] == fam].copy()
    df_neg = sample_negatives(df_all, fam, len(df_pos), neg_ratio)
    df_mix = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)
    df_mix["KinaseLabel"] = df_mix["KIN_GENE"].astype(str).str.upper()
    df_mix.loc[df_mix["Family"] != fam, "KinaseLabel"] = "[NOT_FAM]"

    if backbone.lower() == "aa_onehot":
        embeds = embed_aa(df_mix, windows, fam_dir / "bad_sites_center.csv")
        model_windows = windows
    else:
        embeds = embed_esm(df_mix, windows, cache_dir, backbone, repr_mode=esm_repr_mode, cache_dtype=esm_cache_dtype)
        model_windows = [embeds[L].shape[1] for L in windows]
    Xs = [embeds[L] for L in windows]

    log(f"[Stage-2][{fam}] X branch shapes: {[x.shape for x in Xs]}")
    log(f"[Stage-2][{fam}] time_dims: {[x.shape[1] for x in Xs]}")

    le = LabelEncoder();
    y = le.fit_transform(df_mix["KinaseLabel"].tolist());
    kin_classes = le.classes_.tolist()

    idx = np.arange(len(y))
    strat = y if all(np.bincount(y) > 1) else None
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strat)
    Xtr = [Xi[tr_idx] for Xi in Xs];
    Xva = [Xi[va_idx] for Xi in Xs]
    ytr, yva = y[tr_idx], y[va_idx]

    if use_smote:
        Xtr, ytr = smote_interpolate(Xtr, ytr)
    class_weight = build_class_weight(ytr, num_classes=len(kin_classes)) if use_class_weight else None

    time_dims = [Xi.shape[1] for Xi in Xs]
    use_mlp = all(t == 1 for t in time_dims)

    if use_mlp:
        log(f"[Stage-2][{fam}] Using MLP head (pooled embeddings detected).")
        model = build_multibranch_mlp(len(kin_classes), [Xi.shape[-1] for Xi in Xs], model_windows, ch, drop)
    else:
        log(f"[Stage-2][{fam}] Using Conv head (sequence embeddings detected).")
        model = build_multibranch(len(kin_classes), [Xi.shape[-1] for Xi in Xs], model_windows, ch, drop)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=ce(0.05), metrics=["accuracy"])

    # Manual training loop to avoid GPU/CPU copy issues in some environments
    best = (-1.0, None)
    n_tr = Xtr[0].shape[0]
    idx_all = np.arange(n_tr)

    for ep in range(1, epochs + 1):
        np.random.shuffle(idx_all)
        n_batches = int(math.ceil(n_tr / batch_size))
        for bi in range(n_batches):
            b_idx = idx_all[bi * batch_size:(bi + 1) * batch_size]
            if len(b_idx) == 0:
                continue
            batch_X = [Xi[b_idx] for Xi in Xtr]
            batch_y = ytr[b_idx]
            if use_class_weight and class_weight is not None and len(class_weight) > 0:
                sw = np.array([class_weight[int(c)] for c in batch_y], dtype="float32")
            else:
                sw = None
            if sw is not None:
                model.train_on_batch(batch_X, batch_y, sample_weight=sw)
            else:
                model.train_on_batch(batch_X, batch_y)

        logits = model.predict(Xva, batch_size=256, verbose=0)
        f1_w = f1_score(yva, np.argmax(logits, 1), average="weighted", zero_division=0)
        log(f"[Stage-2][{fam}] Epoch {ep}/{epochs}, val_weighted_f1={f1_w:.4f}, best={best[0]:.4f}")
        if f1_w > best[0]:
            best = (f1_w, model.get_weights())
    if best[1] is not None:
        model.set_weights(best[1])

    # Per-family kinase evaluation (excluding NOT_FAM), same metrics as v16
    logits_va = model.predict(Xva, batch_size=256, verbose=0)
    report_dir = fam_dir / "val"
    eval_multiclass_block(yva, logits_va, kin_classes, report_dir, prefix="kinase")
    return model, le, kin_classes


def infer_family(model_fam: Model, Xs: List[np.ndarray]) -> np.ndarray:
    return model_fam.predict(Xs, batch_size=256, verbose=0)


def infer_kinase_gated(
        df_te: pd.DataFrame,
        fam_names: List[str],
        fam2model: Dict[str, tuple],
        windows: List[int],
        backbone: str,
        cache_dir: Path,
        P_fam: np.ndarray,
        topk=3,
        soft_gate=True,
        esm_repr_mode: str = "pooled",
        esm_cache_dtype: str = "float16",
) -> tuple:
    # build global kinase label list
    all_kin = []
    for fam, (mdl, le, kin_classes) in fam2model.items():
        for k in kin_classes:
            if k == "[NOT_FAM]":
                continue
            all_kin.append(f"{fam}::{k}")
    all_kin = sorted(set(all_kin))
    col_index = {k: i for i, k in enumerate(all_kin)}

    # Embedding (fully backbone-driven)
    if backbone.lower() == "aa_onehot":
        embeds = embed_aa(df_te, windows, None)
    else:
        embeds = embed_esm(
            df_te,
            windows,
            cache_dir,
            backbone,
            repr_mode=esm_repr_mode,
            cache_dtype=esm_cache_dtype,
        )
    Xs = [embeds[L] for L in windows]

    N = len(df_te)
    K = len(all_kin)
    Y_prob = np.zeros((N, K), dtype=np.float32)

    fam_topk_idx = np.argsort(-P_fam, axis=1)[:, :topk]

    for j, fam in enumerate(fam_names):
        if fam not in fam2model:
            continue
        mdl, le, kin_classes = fam2model[fam]
        logits = mdl.predict(Xs, batch_size=256, verbose=0)
        prob = tf.nn.softmax(logits, axis=1).numpy()  # [N, C_fam]

        rows = np.any(fam_topk_idx == j, axis=1)
        if not np.any(rows):
            continue

        gate = P_fam[rows, j][:, None] if soft_gate else 1.0
        for c_name, col in zip(kin_classes, range(prob.shape[1])):
            if c_name == "[NOT_FAM]":
                continue
            gname = f"{fam}::{c_name}"
            gi = col_index.get(gname, None)
            if gi is None:
                continue
            if isinstance(gate, float):
                Y_prob[rows, gi] = prob[rows, col] * gate
            else:
                Y_prob[rows, gi] = prob[rows, col] * gate.squeeze()

    return Y_prob, all_kin


# -------------------- CLI --------------------

# ================= YAML config loader =================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a top-level mapping (YAML dict).")
    return cfg


from pathlib import Path
from typing import List, Dict, Union


def build_seq_map_from_tsvs(tsv_paths: List[Union[str, Path]]) -> Dict[str, str]:
    if not tsv_paths:
        raise ValueError(
            "predict_seq_tsvs is not set in config. "
            "Please provide UniProt TSV files under predict.sequence_lookup_tsvs."
        )

    log(f"[Predict] Loading reference sequences from UniProt TSVs: {tsv_paths}")
    seq_map = load_uniprot_tsvs(tsv_paths)
    log(f"[Predict] Reference sequences for {len(seq_map)} proteins ready (TSV only).")
    return seq_map


def load_uniprot_tsvs(tsv_paths: List[Union[str, Path]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    for p in tsv_paths:
        p = Path(p)
        if not p.exists():
            log(f"[Predict] [WARN] UniProt TSV not found: {p}")
            continue

        # sep=None + engine='python' 自动推断分隔符（逗号 / 制表符），并处理引号
        df = pd.read_csv(p, sep=None, engine="python", comment="#")

        # 兼容不同列名
        entry_col = find_col(df, ["Entry", "ACC", "ACCESSION", "Entry Name"])
        seq_col = find_col(df, ["Sequence", "SEQ"])

        df2 = df[[entry_col, seq_col]].dropna().copy()

        # 去掉首尾空格/引号，统一大写
        df2[entry_col] = (
            df2[entry_col]
            .astype(str)
            .str.strip()
            .str.strip('"')
            .str.strip()
        )
        df2[seq_col] = (
            df2[seq_col]
            .astype(str)
            .str.strip()
            .str.strip('"')
            .str.upper()
        )

        for acc, seq in zip(df2[entry_col], df2[seq_col]):
            if acc not in mapping:
                mapping[acc] = seq

    return mapping


# ==== helper: auto load csv / excel for raw predict ====
def load_table_auto(path_str: str):
    """Load raw table from CSV or Excel automatically."""
    import pandas as pd

    p = str(path_str)
    lower = p.lower()

    # CSV
    if lower.endswith(".csv"):
        return pd.read_csv(p)

    # TSV
    if lower.endswith(".tsv"):
        return pd.read_csv(p, sep="\t")

    # Excel family
    if lower.endswith((".xls", ".xlsx", ".xlsm", ".xlsb", ".ods")):
        try:
            return pd.read_excel(p, engine="openpyxl")
        except Exception:
            return pd.read_excel(p)

    # unknown extension: try excel then csv
    try:
        return pd.read_excel(p)
    except Exception:
        return pd.read_csv(p)


# def run_predict(args, outdir: Path, cache_dir: Path, windows: List[int]):
#     """
#     Prediction-only pipeline:
#       - read raw excel/csv (Substrate_ACC, sites)
#       - use UniProt TSVs to fetch SEQUENCE
#       - run stage-1 family + stage-2 kinase experts with gating
#       - write CSV with columns: GENE, psite (e.g. Q9D0E3_S165)
#
#     Extra outputs in the same folder as predict_output_csv:
#       - raw_input_skipped_no_seq.csv
#       - raw_input_bad_site.csv
#       - raw_input_multi_site.csv
#     """
#
#     # 1) 只用 UniProt TSV 构建 SUB_ACC_ID/Entry -> SEQUENCE 映射
#     seq_map = build_seq_map_from_tsvs(getattr(args, "predict_seq_tsvs", []))
#
#     # 2) 读 raw 输入（可以是 csv/xlsx），至少要有 Substrate_ACC 和 sites 两列
#     if args.predict_raw_excel is None:
#         raise ValueError("predict_raw_excel is not set in config (predict.raw_input_excel).")
#
#     log(f"[Predict] Reading raw excel: {args.predict_raw_excel}")
#     if "load_table_auto" in globals():
#         df_raw = load_table_auto(args.predict_raw_excel)
#     else:
#         df_raw = pd.read_excel(args.predict_raw_excel)
#
#     # patch ID
#     acc_like_cols = {"Substrate_ACC", "SUB_ACC_ID", "ACC", "Protein"}
#     site_like_cols = {"sites", "SUB_MOD_RSD", "SITE", "MOD_RSD"}
#
#     has_acc_like = any(c in df_raw.columns for c in acc_like_cols)
#     has_site_like = any(c in df_raw.columns for c in site_like_cols)
#
#     if (not has_acc_like or not has_site_like):
#         # try ID / psite first
#         id_candidates = [c for c in ["ID", "id", "psite", "PSITE"] if c in df_raw.columns]
#         if id_candidates:
#             id_col = id_candidates[0]
#             log(f"[Predict] No explicit acc/sites columns found, "
#                 f"using '{id_col}' as ACC_SITE (e.g. P38159_S208) and splitting it.")
#
#             # 拆成两列：蛋白质 ID + 位点字符串
#             split_df = df_raw[id_col].astype(str).str.strip().str.split("_", n=1, expand=True)
#             if split_df.shape[1] == 2:
#                 df_raw["Substrate_ACC"] = split_df[0]
#                 df_raw["sites"] = split_df[1]
#             else:
#                 log(f"[Predict] WARNING: some rows in {id_col} cannot be split by '_' into ACC and site; "
#                     f"they will be treated as bad_site later.")
#         else:
#             log("[Predict] No acc/sites columns and no ID/psite column found; "
#                 "will still try find_col but it may fail.")
#
#     acc_col = find_col(df_raw, ["Substrate_ACC", "SUB_ACC_ID", "ACC", "Protein"])
#     site_col = find_col(df_raw, ["sites", "SUB_MOD_RSD", "SITE", "MOD_RSD"])
#
#     rows = []
#     skipped_no_seq = 0
#     skipped_bad_site = 0
#
#     # 额外导出：没有 sequence、bad site、以及 multisite 的原始行
#     no_seq_rows = []
#     bad_site_rows = []
#     multi_site_rows = []
#
#     for _, row in df_raw.iterrows():
#         acc = str(row[acc_col]).strip()  # 保留完整 ID，比如 P28028 和 P28028-1 区分对待
#
#         # ---- 没有 sequence 的样本：整行跳过 ----
#         if acc not in seq_map:
#             skipped_no_seq += 1
#             d = row.to_dict()
#             d["_reason"] = "no_sequence_match"
#             no_seq_rows.append(d)
#             continue
#
#         seq = seq_map[acc]
#
#         # ---- 拆分 sites，支持 S430,T431 / S430 T431 / S430;T431 等 ----
#         raw_site_full = str(row[site_col])
#         tokens = [t for t in re.split(r"[;,\s]+", raw_site_full.strip()) if t]
#
#         # 记录“同一行给了多个 site”的情况，但这些 site 仍然会全部展开参与预测
#         if len(tokens) > 1:
#             d = row.to_dict()
#             d["_parsed_sites"] = ";".join(tokens)
#             multi_site_rows.append(d)
#
#         # 如果整格是空的，也算 bad site
#         if len(tokens) == 0:
#             skipped_bad_site += 1
#             d = row.to_dict()
#             d["_reason"] = "bad_site_format"
#             d["_raw_site"] = ""
#             bad_site_rows.append(d)
#             continue
#
#         # ---- 关键：multi-site 展开成多个样本 ----
#         for raw_site in tokens:
#             ok, aa, pos = site_ok(raw_site)
#
#             # 格式不合法：不是 S/T/Y + 数字
#             if not ok:
#                 skipped_bad_site += 1
#                 d = row.to_dict()
#                 d["_reason"] = "bad_site_format"
#                 d["_raw_site"] = raw_site
#                 bad_site_rows.append(d)
#                 continue
#
#             # 位点位置越界（通常是 ID 和序列不匹配时出现）
#             if pos < 1 or pos > len(seq):
#                 skipped_bad_site += 1
#                 d = row.to_dict()
#                 d["_reason"] = "site_out_of_range"
#                 d["_raw_site"] = raw_site
#                 d["_seq_len"] = len(seq)
#                 bad_site_rows.append(d)
#                 continue
#
#             sub_mod_rsd = f"{aa}{pos}"
#             rows.append({
#                 "SUB_ACC_ID": acc,
#                 "SUB_MOD_RSD": sub_mod_rsd,
#                 "SEQUENCE": seq,
#                 "POS": pos,
#             })
#
#     if not rows:
#         raise ValueError("[Predict] No valid rows after sequence + site matching.")
#
#     df_pred = pd.DataFrame(rows)
#     df_pred["ID"] = df_pred["SUB_ACC_ID"] + "_" + df_pred["SUB_MOD_RSD"]
#     log(f"[Predict] Total valid sites for prediction: {len(df_pred)} "
#         f"(skipped_no_seq={skipped_no_seq}, skipped_bad_site={skipped_bad_site})")
#
#     # 3) 生成 embedding（aa_onehot 或 ESM）
#     if args.backbone.lower() == "aa_onehot":
#         embeds = embed_aa(df_pred, windows, outdir / "predict_bad_sites_center.csv")
#         model_windows = windows
#     else:
#         embeds = embed_esm(df_pred, windows, cache_dir, args.backbone, repr_mode=args.esm_repr_mode, cache_dtype=args.esm_cache_dtype)
#         model_windows = [embeds[L].shape[1] for L in windows]
#
#     Xs = [embeds[L] for L in windows]
#     token_dims = [embeds[L].shape[-1] for L in windows]
#
#     # 4) 加载 stage-1 family 模型
#     stage1_dir = outdir / "stage1_family"
#     fam_classes_path = stage1_dir / "family_classes.npy"
#     fam_weights_path = stage1_dir / "family_model.weights.h5"
#     if not fam_classes_path.exists() or not fam_weights_path.exists():
#         raise FileNotFoundError(
#             f"[Predict] Cannot find stage-1 files in {stage1_dir}. "
#             f"Please train first (mode=train)."
#         )
#
#     fam_names = np.load(fam_classes_path, allow_pickle=True).tolist()
#     fam_model = build_multibranch(
#         n_classes=len(fam_names),
#         token_dims=token_dims,
#         windows=model_windows,
#         ch=args.stage1_ch,
#         drop=args.stage1_drop,
#     )
#     fam_model.load_weights(fam_weights_path)
#     log("[Predict] Stage-1 family model loaded.")
#
#     logits_fam = fam_model.predict(Xs, batch_size=256, verbose=0)
#     P_fam = tf.nn.softmax(logits_fam, axis=1).numpy()
#
#     # 5) 加载每个 family 的 stage-2 kinase expert
#     fam2model = {}
#     for fam in fam_names:
#         fam_dir = outdir / "stage2_kinase" / safe_dirname(fam)
#         kin_cls_path = fam_dir / "kinase_classes.npy"
#         kin_w_path = fam_dir / "kinase_expert.weights.h5"
#         if not kin_cls_path.exists() or not kin_w_path.exists():
#             log(f"[Predict] Skip family {fam}: missing {kin_cls_path.name} or {kin_w_path.name}")
#             continue
#
#         kin_classes = np.load(kin_cls_path, allow_pickle=True).tolist()
#         if len(kin_classes) == 0:
#             continue
#
#         mdl = build_multibranch(
#             n_classes=len(kin_classes),
#             token_dims=token_dims,
#             windows=model_windows,
#             ch=args.stage2_ch,
#             drop=args.stage2_drop,
#         )
#         mdl.load_weights(kin_w_path)
#         fam2model[fam] = (mdl, None, kin_classes)
#
#     if not fam2model:
#         raise ValueError("[Predict] No stage-2 kinase experts loaded. Train first.")
#
#     # 6) 两阶段 gating 得到全局 kinase 概率矩阵
#     Y_prob_kin, all_kin_cols = infer_kinase_gated(
#         df_pred,
#         fam_names,
#         fam2model,
#         windows,
#         args.backbone,
#         cache_dir,
#         P_fam=P_fam,
#         topk=args.topk_family,
#         soft_gate=args.soft_gate,
#         esm_repr_mode=args.esm_repr_mode,
#         esm_cache_dtype=args.esm_cache_dtype,
#     )
#
#     if Y_prob_kin.shape[1] == 0:
#         raise ValueError("[Predict] Empty kinase probability matrix.")
#
#     top_idx = np.argmax(Y_prob_kin, axis=1)
#     top_global = [all_kin_cols[i] for i in top_idx]
#     genes = [g.split("::", 1)[1] if "::" in g else g for g in top_global]
#
#     # 7) 写最终预测：GENE & psite
#     df_out = pd.DataFrame({
#         "GENE": genes,
#         "psite": df_pred["SUB_ACC_ID"].astype(str).str.strip() + "_" +
#                  df_pred["SUB_MOD_RSD"].astype(str).str.strip(),
#     })
#
#     out_csv = Path(args.predict_output_csv)
#     out_dir = out_csv.parent
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     df_out.to_csv(out_csv, index=False)
#     log(f"[Predict] Saved {len(df_out)} predictions to: {out_csv}")
#
#     # 额外：导出三种“问题样本”的原始行
#     if len(no_seq_rows) > 0:
#         df_no = pd.DataFrame(no_seq_rows)
#         no_path = out_dir / "raw_input_skipped_no_seq.csv"
#         df_no.to_csv(no_path, index=False)
#         log(f"[Predict] Saved {len(df_no)} raw rows with no sequence match to: {no_path}")
#
#     if len(bad_site_rows) > 0:
#         df_bad = pd.DataFrame(bad_site_rows)
#         bad_path = out_dir / "raw_input_bad_site.csv"
#         df_bad.to_csv(bad_path, index=False)
#         log(f"[Predict] Saved {len(df_bad)} raw rows with bad sites to: {bad_path}")
#
#     if len(multi_site_rows) > 0:
#         df_multi = pd.DataFrame(multi_site_rows)
#         multi_path = out_dir / "raw_input_multi_site.csv"
#         df_multi.to_csv(multi_path, index=False)
#         log(f"[Predict] Saved {len(df_multi)} raw rows with multiple sites per row to: {multi_path}")
def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def load_best_f1_thresholds_from_pr_curves(pr_csv_path: Path) -> dict:
    """
    Read a PR-curve CSV with columns: label, precision, recall, threshold
    and return best threshold per label by maximizing F1 = 2PR/(P+R).

    Notes:
      - Some PR curve exports append a last row with threshold=1.0 (no model threshold);
        we still consider it but it rarely wins.
      - If a label has no valid rows, it's omitted.
    """
    pr_csv_path = Path(pr_csv_path)
    if not pr_csv_path.exists():
        return {}

    df = pd.read_csv(pr_csv_path)
    need_cols = {"label", "precision", "recall", "threshold"}
    if not need_cols.issubset(df.columns):
        return {}

    # compute F1 safely
    p = df["precision"].astype(float).replace([np.inf, -np.inf], np.nan)
    r = df["recall"].astype(float).replace([np.inf, -np.inf], np.nan)
    denom = (p + r)
    f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    df = df.assign(_f1=f1)

    best = {}
    for lab, g in df.groupby("label"):
        g2 = g.dropna(subset=["threshold", "_f1"])
        if len(g2) == 0:
            continue
        # pick max f1; if tie, prefer higher recall (slightly safer), then lower threshold
        g2 = g2.sort_values(["_f1", "recall", "threshold"], ascending=[False, False, True])
        thr = _safe_float(g2.iloc[0]["threshold"], default=None)
        if thr is None:
            continue
        best[str(lab)] = float(thr)
    return best


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        try:
            p = Path(p)
            if p.exists():
                return p
        except Exception:
            continue
    return None


def guess_pr_curve_path(outdir: Path, which: str) -> Path | None:
    """Try to find a PR-curve CSV inside an outdir. which in {'kinase','family'}"""
    outdir = Path(outdir)
    which = str(which).lower().strip()
    cand = []
    if which == "kinase":
        cand += [
            outdir / "eval" / "kinase_pr_curves.csv",
            outdir / "stage2_kinase" / "kinase_pr_curves.csv",
            outdir / "kinase_pr_curves.csv",
        ]
        # any nested
        cand += list(outdir.glob("**/kinase_pr_curves.csv"))[:10]
    else:
        cand += [
            outdir / "eval" / "family_pr_curves.csv",
            outdir / "stage1_family" / "family_pr_curves.csv",
            outdir / "family_pr_curves.csv",
        ]
        cand += list(outdir.glob("**/family_pr_curves.csv"))[:10]
    return _find_first_existing(cand)


def run_predict(args, outdir: Path, cache_dir: Path, windows: List[int]):
    """
    Prediction-only pipeline:
      - read raw excel/csv (Substrate_ACC, sites)
      - use UniProt TSVs to fetch SEQUENCE
      - run stage-1 family + stage-2 kinase experts with gating
      - write CSV with columns: GENE, psite (e.g. Q9D0E3_S165)

    Extra outputs in the same folder as predict_output_csv:
      - raw_input_skipped_no_seq.csv
      - raw_input_bad_site.csv
      - raw_input_multi_site.csv

    NEW (scheme2: threshold + topk) outputs:
      - <stem>_pairs.csv : psite, kinase, family_score, kinase_score, final_score
      - <stem>_family_kinase_num_psite.csv : family, kinase, num_psite
      - <stem>_psite_summary.csv : psite-level summary for debugging
    """

    # -------------------------
    # (A) parse scheme2 configs
    # -------------------------
    # -------------------------
    # (A) parse scheme2 configs (threshold + topK, 2-level thresholds)
    # -------------------------
    # Backward-compatible: old predict_score_threshold now acts as an *optional* final_score floor.
    final_score_min = float(getattr(args, "predict_score_threshold", 0.0))

    # Family threshold
    family_thr_mode = str(getattr(args, "predict_family_threshold_mode", "fixed")).lower().strip()
    family_thr_default = float(getattr(args, "predict_family_threshold", 0.0))
    family_thr_map = {}
    if family_thr_mode in ("auto", "auto_f1", "f1"):
        fam_pr = getattr(args, "predict_family_pr_curves_csv", None)
        fam_pr_path = Path(fam_pr) if fam_pr else guess_pr_curve_path(outdir, "family")
        if fam_pr_path and Path(fam_pr_path).exists():
            family_thr_map = load_best_f1_thresholds_from_pr_curves(Path(fam_pr_path))
            log(f"[Predict] Loaded {len(family_thr_map)} family thresholds from PR curves: {fam_pr_path}")
        else:
            log("[Predict] [WARN] family_threshold_mode=auto_f1 but family_pr_curves.csv not found; "
                "falling back to fixed family threshold.")

    # Kinase threshold
    kinase_thr_mode = str(getattr(args, "predict_kinase_threshold_mode", "auto_f1")).lower().strip()
    kinase_thr_default = float(getattr(args, "predict_kinase_threshold", 0.0))
    kinase_thr_map = {}
    if kinase_thr_mode in ("auto", "auto_f1", "f1"):
        kin_pr = getattr(args, "predict_kinase_pr_curves_csv", None)
        kin_pr_path = Path(kin_pr) if kin_pr else guess_pr_curve_path(outdir, "kinase")
        if kin_pr_path and Path(kin_pr_path).exists():
            kinase_thr_map = load_best_f1_thresholds_from_pr_curves(Path(kin_pr_path))
            log(f"[Predict] Loaded {len(kinase_thr_map)} kinase thresholds from PR curves: {kin_pr_path}")
        else:
            log("[Predict] [WARN] kinase_threshold_mode=auto_f1 but kinase_pr_curves.csv not found; "
                "falling back to fixed kinase threshold.")

    # TopK safety net
    topk_kin_per_family = int(getattr(args, "predict_topk_kinase_per_family", getattr(args, "predict_topk_kinase", 5)))
    if topk_kin_per_family <= 0:
        topk_kin_per_family = 1
    max_total = int(getattr(args, "predict_max_candidates_total", 0))

    # 1) 只用 UniProt TSV 构建 SUB_ACC_ID/Entry -> SEQUENCE 映射
    seq_map = build_seq_map_from_tsvs(getattr(args, "predict_seq_tsvs", []))

    # 2) 读 raw 输入（可以是 csv/xlsx），至少要有 Substrate_ACC 和 sites 两列
    if args.predict_raw_excel is None:
        raise ValueError("predict_raw_excel is not set in config (predict.raw_input_excel).")

    log(f"[Predict] Reading raw excel: {args.predict_raw_excel}")
    if "load_table_auto" in globals():
        df_raw = load_table_auto(args.predict_raw_excel)
    else:
        df_raw = pd.read_excel(args.predict_raw_excel)

    # patch ID
    acc_like_cols = {"Substrate_ACC", "SUB_ACC_ID", "ACC", "Protein"}
    site_like_cols = {"sites", "SUB_MOD_RSD", "SITE", "MOD_RSD"}

    has_acc_like = any(c in df_raw.columns for c in acc_like_cols)
    has_site_like = any(c in df_raw.columns for c in site_like_cols)

    if (not has_acc_like or not has_site_like):
        # try ID / psite first
        id_candidates = [c for c in ["ID", "id", "psite", "PSITE"] if c in df_raw.columns]
        if id_candidates:
            id_col = id_candidates[0]
            log(f"[Predict] No explicit acc/sites columns found, "
                f"using '{id_col}' as ACC_SITE (e.g. P38159_S208) and splitting it.")

            # 拆成两列：蛋白质 ID + 位点字符串
            split_df = df_raw[id_col].astype(str).str.strip().str.split("_", n=1, expand=True)
            if split_df.shape[1] == 2:
                df_raw["Substrate_ACC"] = split_df[0]
                df_raw["sites"] = split_df[1]
            else:
                log(f"[Predict] WARNING: some rows in {id_col} cannot be split by '_' into ACC and site; "
                    f"they will be treated as bad_site later.")
        else:
            log("[Predict] No acc/sites columns and no ID/psite column found; "
                "will still try find_col but it may fail.")

    acc_col = find_col(df_raw, ["Substrate_ACC", "SUB_ACC_ID", "ACC", "Protein"])
    site_col = find_col(df_raw, ["sites", "SUB_MOD_RSD", "SITE", "MOD_RSD"])

    rows = []
    skipped_no_seq = 0
    skipped_bad_site = 0

    # 额外导出：没有 sequence、bad site、以及 multisite 的原始行
    no_seq_rows = []
    bad_site_rows = []
    multi_site_rows = []

    for _, row in df_raw.iterrows():
        acc = str(row[acc_col]).strip()  # 保留完整 ID，比如 P28028 和 P28028-1 区分对待

        # ---- 没有 sequence 的样本：整行跳过 ----
        if acc not in seq_map:
            skipped_no_seq += 1
            d = row.to_dict()
            d["_reason"] = "no_sequence_match"
            no_seq_rows.append(d)
            continue

        seq = seq_map[acc]

        # ---- 拆分 sites，支持 S430,T431 / S430 T431 / S430;T431 等 ----
        raw_site_full = str(row[site_col])
        tokens = [t for t in re.split(r"[;,\s]+", raw_site_full.strip()) if t]

        # 记录“同一行给了多个 site”的情况，但这些 site 仍然会全部展开参与预测
        if len(tokens) > 1:
            d = row.to_dict()
            d["_parsed_sites"] = ";".join(tokens)
            multi_site_rows.append(d)

        # 如果整格是空的，也算 bad site
        if len(tokens) == 0:
            skipped_bad_site += 1
            d = row.to_dict()
            d["_reason"] = "bad_site_format"
            d["_raw_site"] = ""
            bad_site_rows.append(d)
            continue

        # ---- 关键：multi-site 展开成多个样本 ----
        for raw_site in tokens:
            ok, aa, pos = site_ok(raw_site)

            # 格式不合法：不是 S/T/Y + 数字
            if not ok:
                skipped_bad_site += 1
                d = row.to_dict()
                d["_reason"] = "bad_site_format"
                d["_raw_site"] = raw_site
                bad_site_rows.append(d)
                continue

            # 位点位置越界（通常是 ID 和序列不匹配时出现）
            if pos < 1 or pos > len(seq):
                skipped_bad_site += 1
                d = row.to_dict()
                d["_reason"] = "site_out_of_range"
                d["_raw_site"] = raw_site
                d["_seq_len"] = len(seq)
                bad_site_rows.append(d)
                continue

            sub_mod_rsd = f"{aa}{pos}"
            rows.append({
                "SUB_ACC_ID": acc,
                "SUB_MOD_RSD": sub_mod_rsd,
                "SEQUENCE": seq,
                "POS": pos,
            })

    if not rows:
        raise ValueError("[Predict] No valid rows after sequence + site matching.")

    df_pred = pd.DataFrame(rows)
    df_pred["ID"] = df_pred["SUB_ACC_ID"] + "_" + df_pred["SUB_MOD_RSD"]
    log(f"[Predict] Total valid sites for prediction: {len(df_pred)} "
        f"(skipped_no_seq={skipped_no_seq}, skipped_bad_site={skipped_bad_site})")

    # 3) 生成 embedding（aa_onehot 或 ESM）
    if args.backbone.lower() == "aa_onehot":
        embeds = embed_aa(df_pred, windows, outdir / "predict_bad_sites_center.csv")
        model_windows = windows
    else:
        embeds = embed_esm(
            df_pred, windows, cache_dir, args.backbone,
            repr_mode=args.esm_repr_mode,
            cache_dtype=args.esm_cache_dtype
        )
        model_windows = [embeds[L].shape[1] for L in windows]

    Xs = [embeds[L] for L in windows]
    token_dims = [embeds[L].shape[-1] for L in windows]

    # 4) 加载 stage-1 family 模型
    stage1_dir = outdir / "stage1_family"
    fam_classes_path = stage1_dir / "family_classes.npy"
    fam_weights_path = stage1_dir / "family_model.weights.h5"
    if not fam_classes_path.exists() or not fam_weights_path.exists():
        raise FileNotFoundError(
            f"[Predict] Cannot find stage-1 files in {stage1_dir}. "
            f"Please train first (mode=train)."
        )

    fam_names = np.load(fam_classes_path, allow_pickle=True).tolist()
    fam_model = build_multibranch(
        n_classes=len(fam_names),
        token_dims=token_dims,
        windows=model_windows,
        ch=args.stage1_ch,
        drop=args.stage1_drop,
    )
    fam_model.load_weights(fam_weights_path)
    log("[Predict] Stage-1 family model loaded.")

    logits_fam = fam_model.predict(Xs, batch_size=256, verbose=0)
    P_fam = tf.nn.softmax(logits_fam, axis=1).numpy()

    # 5) 加载每个 family 的 stage-2 kinase expert
    fam2model = {}
    for fam in fam_names:
        fam_dir = outdir / "stage2_kinase" / safe_dirname(fam)
        kin_cls_path = fam_dir / "kinase_classes.npy"
        kin_w_path = fam_dir / "kinase_expert.weights.h5"
        if not kin_cls_path.exists() or not kin_w_path.exists():
            log(f"[Predict] Skip family {fam}: missing {kin_cls_path.name} or {kin_w_path.name}")
            continue

        kin_classes = np.load(kin_cls_path, allow_pickle=True).tolist()
        if len(kin_classes) == 0:
            continue

        mdl = build_multibranch(
            n_classes=len(kin_classes),
            token_dims=token_dims,
            windows=model_windows,
            ch=args.stage2_ch,
            drop=args.stage2_drop,
        )
        mdl.load_weights(kin_w_path)
        fam2model[fam] = (mdl, None, kin_classes)

    if not fam2model:
        raise ValueError("[Predict] No stage-2 kinase experts loaded. Train first.")

    # 6) 两阶段 gating 得到全局 kinase 概率矩阵
    Y_prob_kin, all_kin_cols = infer_kinase_gated(
        df_pred,
        fam_names,
        fam2model,
        windows,
        args.backbone,
        cache_dir,
        P_fam=P_fam,
        topk=args.topk_family,
        soft_gate=args.soft_gate,
        esm_repr_mode=args.esm_repr_mode,
        esm_cache_dtype=args.esm_cache_dtype,
    )

    if Y_prob_kin.shape[1] == 0:
        raise ValueError("[Predict] Empty kinase probability matrix.")

    # --------------------------------------------
    # 7) 原版输出：每个 psite 只输出 top1 kinase gene
    # --------------------------------------------
    top_idx = np.argmax(Y_prob_kin, axis=1)
    top_global = [all_kin_cols[i] for i in top_idx]
    genes = [g.split("::", 1)[1] if "::" in g else g for g in top_global]

    psites = (
            df_pred["SUB_ACC_ID"].astype(str).str.strip()
            + "_"
            + df_pred["SUB_MOD_RSD"].astype(str).str.strip()
    )

    df_out = pd.DataFrame({
        "GENE": genes,
        "psite": psites,
    })

    out_csv = Path(args.predict_output_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(out_csv, index=False)
    log(f"[Predict] Saved {len(df_out)} predictions to: {out_csv}")

    # ---------------------------------------------------------
    # 8) NEW: scheme2（阈值 + topk）输出多候选 pairs 表
    #    columns: psite, kinase, family_score, kinase_score, final_score
    # ---------------------------------------------------------
    # fam2idx (kept below)

    pair_rows = []
    n = Y_prob_kin.shape[0]

    fam2idx = {f: i for i, f in enumerate(fam_names)}

    # Build a family -> global kinase column indices map (family::kinase)
    fam_to_kin_idx: dict[str, list[int]] = {}
    for gi, col in enumerate(all_kin_cols):
        if "::" not in col:
            continue
        fam, kin = col.split("::", 1)
        fam_to_kin_idx.setdefault(fam, []).append(gi)

    eps = 1e-12

    # family topK (same gating logic as infer_kinase_gated)
    fam_topk_idx = np.argsort(-P_fam, axis=1)[:, : int(getattr(args, "topk_family", 3))]

    for i in range(n):
        psite_i = str(psites.iloc[i])

        # -------------------------
        # Step 1: choose family candidates (topK + family threshold, with safety net)
        # -------------------------
        fam_candidates = []
        for fj in fam_topk_idx[i]:
            fam = str(fam_names[int(fj)])
            fam_score = float(P_fam[i, int(fj)])
            thr_f = float(family_thr_map.get(fam, family_thr_default))
            if fam_score >= thr_f:
                fam_candidates.append((int(fj), fam, fam_score))

        # safety: never empty
        if len(fam_candidates) == 0:
            fj0 = int(fam_topk_idx[i][0])
            fam0 = str(fam_names[fj0])
            fam_candidates = [(fj0, fam0, float(P_fam[i, fj0]))]

        # -------------------------
        # Step 2: within each selected family, apply kinase threshold on P(kinase|family)
        #         + keep TopK-per-family as safety net
        # -------------------------
        cand_rows = []
        cand_seen = set()

        for fj, fam, fam_score in fam_candidates:
            kin_idxs = fam_to_kin_idx.get(fam, [])
            if not kin_idxs:
                continue

            final_scores = Y_prob_kin[i, kin_idxs].astype(float)

            # conditional score reconstruction
            if getattr(args, "soft_gate", True) and fam_score > eps:
                kin_scores = final_scores / (fam_score + eps)  # P(kin|fam)
            else:
                kin_scores = final_scores  # already conditional (or best-effort)

            # sort by final score desc
            order = np.argsort(-final_scores)
            topk_local = order[: min(topk_kin_per_family, len(order))]

            # threshold pass set (label-specific threshold)
            pass_local = []
            for oi in order:
                gi = kin_idxs[int(oi)]
                col = all_kin_cols[gi]
                thr_k = float(kinase_thr_map.get(str(col), kinase_thr_default))
                if kin_scores[int(oi)] >= thr_k:
                    pass_local.append(int(oi))

            # safety: always include topk_local
            kept_local = set(pass_local) | set(map(int, topk_local))
            if len(kept_local) == 0 and len(order) > 0:
                kept_local = {int(order[0])}

            for oi in kept_local:
                gi = kin_idxs[int(oi)]
                col = all_kin_cols[gi]
                if col in cand_seen:
                    continue
                cand_seen.add(col)

                if "::" in col:
                    fam2, kin = col.split("::", 1)
                else:
                    fam2, kin = fam, col

                final_score = float(Y_prob_kin[i, gi])
                # recompute kinase_score aligned with soft_gate flag
                if getattr(args, "soft_gate", True) and fam_score > eps:
                    kinase_score = float(final_score / (fam_score + eps))
                else:
                    kinase_score = float(final_score)

                # optional final-score floor (backward-compatible)
                if final_score_min > 0 and final_score < final_score_min:
                    continue

                sqrt_score = math.sqrt(final_score) if final_score > 0 else 0.0
                cand_rows.append({
                    "psite": psite_i,
                    "kinase": str(kin),
                    "family_score": float(fam_score),
                    "kinase_score": float(kinase_score),
                    "final_score": float(sqrt_score),
                    "raw_prob_product": float(final_score),
                    "family": str(fam2),
                })

        # Step 3: optional cap total candidates per psite
        if max_total and max_total > 0 and len(cand_rows) > max_total:
            cand_rows = sorted(cand_rows, key=lambda d: d["final_score"], reverse=True)[: int(max_total)]

        pair_rows.extend(cand_rows)
    df_pairs = pd.DataFrame(pair_rows)

    stem = out_csv.stem
    pairs_path = out_dir / f"{stem}_pairs.csv"
    df_pairs.to_csv(pairs_path, index=False)
    log(f"[Predict] Saved scheme2 pairs table ({len(df_pairs)}) to: {pairs_path}")

    # ---------------------------------------------------------
    # 9) NEW: 第二张表 family, kinase, num_psite（按 psite 去重计数）
    # ---------------------------------------------------------
    if len(df_pairs) > 0:
        df_counts = (
            df_pairs.groupby(["family", "kinase"])["psite"]
            .nunique()
            .reset_index(name="num_psite")
            .sort_values(["num_psite", "family", "kinase"], ascending=[False, True, True])
        )
    else:
        df_counts = pd.DataFrame(columns=["family", "kinase", "num_psite"])

    counts_path = out_dir / f"{stem}_family_kinase_num_psite.csv"
    df_counts.to_csv(counts_path, index=False)
    log(f"[Predict] Saved family-kinase count table ({len(df_counts)}) to: {counts_path}")

    # ---------------------------------------------------------
    # 10) Extra table (recommended): psite-level summary
    # ---------------------------------------------------------
    if len(df_pairs) > 0:
        # top1 per psite within the kept candidates
        df_pairs_sorted = df_pairs.sort_values(["psite", "final_score"], ascending=[True, False])
        top1 = df_pairs_sorted.groupby("psite").head(1).copy()

        df_summary = (
            df_pairs.groupby("psite")
            .agg(
                num_candidates=("kinase", "count"),
                max_final_score=("final_score", "max"),
                mean_final_score=("final_score", "mean"),
            )
            .reset_index()
        )
        df_summary = df_summary.merge(
            top1[["psite", "family", "kinase", "family_score", "kinase_score", "final_score"]]
            .rename(columns={
                "family": "top1_family",
                "kinase": "top1_kinase",
                "family_score": "top1_family_score",
                "kinase_score": "top1_kinase_score",
                "final_score": "top1_final_score",
            }),
            on="psite",
            how="left",
        ).sort_values(["num_candidates", "max_final_score"], ascending=[False, False])
    else:
        df_summary = pd.DataFrame(columns=[
            "psite", "num_candidates", "max_final_score", "mean_final_score",
            "top1_family", "top1_kinase", "top1_family_score", "top1_kinase_score", "top1_final_score"
        ])

    summary_path = out_dir / f"{stem}_psite_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    log(f"[Predict] Saved psite summary table ({len(df_summary)}) to: {summary_path}")

    # 额外：导出三种“问题样本”的原始行
    if len(no_seq_rows) > 0:
        df_no = pd.DataFrame(no_seq_rows)
        no_path = out_dir / "raw_input_skipped_no_seq.csv"
        df_no.to_csv(no_path, index=False)
        log(f"[Predict] Saved {len(df_no)} raw rows with no sequence match to: {no_path}")

    if len(bad_site_rows) > 0:
        df_bad = pd.DataFrame(bad_site_rows)
        bad_path = out_dir / "raw_input_bad_site.csv"
        df_bad.to_csv(bad_path, index=False)
        log(f"[Predict] Saved {len(df_bad)} raw rows with bad sites to: {bad_path}")

    if len(multi_site_rows) > 0:
        df_multi = pd.DataFrame(multi_site_rows)
        multi_path = out_dir / "raw_input_multi_site.csv"
        df_multi.to_csv(multi_path, index=False)
        log(f"[Predict] Saved {len(df_multi)} raw rows with multiple sites per row to: {multi_path}")


def parse_args():
    import argparse
    import yaml
    from pathlib import Path
    from datetime import datetime  # <--- 修改1：导入 datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    cli_args = parser.parse_args()

    # ---- load yaml (self-contained, no external helper) ----
    with open(cli_args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def get_cfg(cfg_dict, dotted_key, default=None):
        """Get nested yaml value by dotted path like 'stage1.epochs'."""
        cur = cfg_dict
        for k in dotted_key.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def get(path, default=None):
        return get_cfg(cfg, path, default)

    ns = argparse.Namespace()

    # ===== paths / data =====
    ns.dataset_csv = get("paths.dataset_csv")
    ns.family_map_csv = get("paths.family_map_csv")

    # <--- 修复：先获取运行模式，再决定是否加时间戳 --->
    base_outdir = get("paths.outdir", "outputs_v26")
    run_mode = str(get("run.mode", "train")).lower()

    if run_mode == "predict":
        # 预测模式：直接使用配置文件里的路径，不加时间戳
        # 这样才能找到已经存在的模型文件夹
        ns.outdir = base_outdir
        print(f"[Config] Predict mode: Using existing output dir: {ns.outdir}")
    else:
        # 训练模式：生成带时间戳的新目录，防止覆盖旧结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ns.outdir = f"{base_outdir}_{timestamp}"
        print(f"[Config] Train mode: Created new output dir: {ns.outdir}")
    # 结果示例: outputs_v26/esm2_t33_UR50D_20251222_103000

    ns.cache_dir = get("paths.cache_dir", "cache_v26")

    # ===== split / families =====
    ns.test_size = float(get("data.test_size", 0.2))
    ns.num_top_families = int(get("data.num_top_families", 10))

    # ===== backbone / windows =====
    ns.backbone = str(get("model.backbone", "aa_onehot"))
    ns.windows = str(get("model.windows", "51,33,15"))

    ns.esm_repr_mode = str(get("model.esm_repr_mode", DEFAULT_ESM_REPR_MODE)).lower().strip()
    ns.esm_cache_dtype = str(get("model.esm_cache_dtype", DEFAULT_ESM_CACHE_DTYPE)).lower().strip()

    # ===== stage-1 =====
    ns.stage1_epochs = int(get("stage1.epochs", 30))
    ns.stage1_ch = int(get("stage1.ch", 256))
    ns.stage1_drop = float(get("stage1.drop", 0.3))
    ns.stage1_lr = float(get("stage1.lr", 2e-4))
    ns.stage1_batch = int(get("stage1.batch_size", 128))
    ns.oversampler_family = str(get("stage1.oversampler_family", "none")).lower()
    ns.use_class_weight_family = bool(get("stage1.use_class_weight_family", True))

    # ===== stage-2 =====
    ns.stage2_epochs = int(get("stage2.epochs", 20))
    ns.stage2_ch = int(get("stage2.ch", 256))
    ns.stage2_drop = float(get("stage2.drop", 0.3))
    ns.stage2_lr = float(get("stage2.lr", 2e-4))
    ns.stage2_batch = int(get("stage2.batch_size", 128))
    ns.oversampler_kinase = str(get("stage2.oversampler_kinase", "none")).lower()
    ns.use_class_weight_kinase = bool(get("stage2.use_class_weight_kinase", True))

    ns.neg_ratio_mode = str(get("stage2.neg_ratio_mode", "fixed")).lower()
    ns.neg_ratio_fixed = float(get("stage2.neg_ratio_fixed", 0.2))
    ns.neg_ratio_minmax = str(get("stage2.neg_ratio_minmax", "0.1,0.5"))

    # ===== gating =====
    ns.topk_family = int(get("gating.topk_family", 3))
    ns.soft_gate = bool(get("gating.soft_gate", True))

    # ===== eval toggles =====
    ns.enable_family_test_eval = bool(get("eval.enable_family_test_eval", True))
    ns.enable_global_kinase_eval = bool(get("eval.enable_global_kinase_eval", True))
    ns.enable_per_family_heatmap = bool(get("eval.enable_per_family_heatmap", True))
    ns.save_run_log = bool(get("eval.save_run_log", True))

    # ===== seed + run mode =====
    ns.random_seed = int(get("run.random_seed", 42))
    ns.run_mode = str(get("run.mode", "train")).lower()

    # ===== predict config =====
    ns.predict_raw_excel = get("predict.raw_input_excel", None)
    ns.predict_seq_csv = get("predict.sequence_lookup_csv", ns.dataset_csv)

    # NEW: optional UniProt TSVs
    ns.predict_seq_tsvs = get("predict.sequence_lookup_tsvs", [])
    if ns.predict_seq_tsvs is None:
        ns.predict_seq_tsvs = []

    # scheme2 (multi-candidate prediction) knobs
    ns.predict_score_threshold = float(get("predict.score_threshold", 0.0))  # optional final_score floor
    ns.predict_topk_kinase = int(get("predict.topk_kinase", 5))  # backward-compatible name
    ns.predict_topk_kinase_per_family = int(get("predict.topk_kinase_per_family", ns.predict_topk_kinase))
    ns.predict_max_candidates_total = int(get("predict.max_candidates_total", 0))

    ns.predict_family_threshold_mode = str(get("predict.family_threshold_mode", "fixed"))
    ns.predict_family_threshold = float(get("predict.family_threshold", 0.0))
    ns.predict_family_pr_curves_csv = get("predict.family_pr_curves_csv", None)

    ns.predict_kinase_threshold_mode = str(get("predict.kinase_threshold_mode", "auto_f1"))
    ns.predict_kinase_threshold = float(get("predict.kinase_threshold", 0.0))
    ns.predict_kinase_pr_curves_csv = get("predict.kinase_pr_curves_csv", None)

    # <--- 修改3：强制将预测输出重定向到带时间戳的文件夹内 --->
    configured_pred_csv = get("predict.output_csv")
    if configured_pred_csv:
        # 提取文件名 (例如 raw_predict_kinase.csv) 并拼接到新的 outdir
        file_name = Path(configured_pred_csv).name
        ns.predict_output_csv = str(Path(ns.outdir) / file_name)
    else:
        # 默认情况
        ns.predict_output_csv = str(Path(ns.outdir) / "raw_predict_kinase.csv")

    ns.config_path = cli_args.config
    return ns


def main():
    args = parse_args()
    outdir = Path(args.outdir);
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir);
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.random_seed)
    # windows = [int(x) for x in args.windows.split(",")]
    w_str = str(args.windows).strip()
    w_str = w_str.strip("[](){}")  # 去掉可能的 [] 括号
    w_str = w_str.replace(" ", "")  # 去掉空格
    windows = [int(x) for x in w_str.split(",") if x != ""]

    run_mode = getattr(args, "run_mode", "train").lower().strip()
    if run_mode == "predict":
        log("==== JUMPTrans hierarchical PREDICTION start ====")
        log(f"Args (from config): {vars(args)}")
        run_predict(args, outdir, cache_dir, windows)
        # Dump run log to text file under outputs for prediction as well
        if getattr(args, "save_run_log", False):
            log_path = outdir / "run_log.txt"
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    for line in RUN_LOG:
                        f.write(line + "\n")
                print(f"[LOG] Run log has been saved to: {log_path}")
            except Exception as e:
                print(f"[LOG] Failed to write run log: {e}")
        return

    log("==== JUMPTrans hierarchical training start ====")
    log(f"Args (from config): {vars(args)}")
    minmax = tuple(float(x) for x in args.neg_ratio_minmax.split(","))

    fam_map = load_family_mapping(args.family_map_csv)
    df_raw = pd.read_csv(args.dataset_csv)
    log(f"[Data] Loaded raw dataset with {len(df_raw)} rows")
    df_all = clean_align(df_raw, fam_map, outdir)
    log(f"[Data] After cleaning/alignment: {len(df_all)} rows")
    df_all.to_csv(outdir / "cleaned_snapshot.csv", index=False)

    # ===== Family / Kinase statistics =====
    fam_counts = df_all["Family"].value_counts().sort_values(ascending=False)
    log(f"[STAT] Total {len(fam_counts)} families in cleaned data")
    for fam, cnt in fam_counts.items():
        n_kin = df_all.loc[df_all["Family"] == fam, "KIN_GENE"].nunique()
        print(f"[STAT]   {fam:20s}  {cnt:6d} sites, {n_kin:4d} kinases")

    # ===== Select Top-K families, merge others into 'Others' =====
    TOP_K = args.num_top_families
    top_fams = fam_counts.index[:TOP_K].tolist()
    print(f"[INFO] Using Top-{TOP_K} families as explicit classes:")
    for fam in top_fams:
        print(f"[INFO]   - {fam}")

    df_all["Family_raw"] = df_all["Family"]
    df_all.loc[~df_all["Family"].isin(top_fams), "Family"] = "Others"

    # 80/20 split (stratified by family)
    y_fam = LabelEncoder().fit_transform(df_all["Family"].tolist())
    strat = y_fam if all(np.bincount(y_fam) > 1) else None
    tr_idx, te_idx = train_test_split(np.arange(len(df_all)), test_size=0.2, random_state=SEED, stratify=strat)
    log(f"[Split] Train size={len(tr_idx)}, Test size={len(te_idx)} (stratified by family)")
    df_tr = df_all.iloc[tr_idx].reset_index(drop=True)
    df_te = df_all.iloc[te_idx].reset_index(drop=True)

    # Stage-1 Family
    use_smote_fam = (args.oversampler_family == "smote")
    log("[Stage-1] Begin training global family classifier ...")
    fam_model, fam_le, fam_names, (Xs_all, y_all, df_tr_all, logits_tr) = train_family(
        df_tr, windows, args.backbone, cache_dir,
        esm_repr_mode=args.esm_repr_mode, esm_cache_dtype=args.esm_cache_dtype,
        use_smote=use_smote_fam, use_class_weight=args.use_class_weight_family,
        ch=args.stage1_ch, drop=args.stage1_drop, lr=args.stage1_lr, epochs=args.stage1_epochs,
        batch_size=args.stage1_batch, out_dir=outdir / "stage1_family"
    )

    # Save Stage-1 family model weights and classes (for later prediction)
    stage1_dir = outdir / "stage1_family"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    fam_model.save_weights(stage1_dir / "family_model.weights.h5")
    np.save(stage1_dir / "family_classes.npy", np.array(fam_names, dtype=object))

    # Family        v16
    if args.backbone.lower() == "aa_onehot":
        embeds_te = embed_aa(df_te, windows, outdir / "bad_sites_center_test.csv")
    else:
        embeds_te = embed_esm(df_te, windows, cache_dir, args.backbone, repr_mode=args.esm_repr_mode,
                              cache_dtype=args.esm_cache_dtype)
    Xs_te = [embeds_te[L] for L in windows]
    logits_te = fam_model.predict(Xs_te, batch_size=256, verbose=0)
    y_te = fam_le.transform(df_te["Family"].tolist())
    eval_multiclass_block(y_te, logits_te, fam_names, outdir / "families_global/test", prefix="family")

    # Stage-2 per-family experts (as v16, with negative-sample strategy)
    fam2model = {}
    log("[Stage-2] Begin training per-family kinase experts ...")
    for fam in fam_names:
        n_pos = int((df_tr_all["Family"] == fam).sum())
        ratio = neg_ratio_value(args.neg_ratio_mode, args.neg_ratio_fixed, n_pos, minmax=minmax)
        log(f"[Stage-2] Training expert for family={fam}, n_pos={n_pos}, neg_ratio={ratio:.3f}")
        mdl, le, kin_classes = train_kinase_expert(
            df_tr_all, fam, windows, args.backbone, cache_dir,
            esm_repr_mode=args.esm_repr_mode, esm_cache_dtype=args.esm_cache_dtype,
            neg_ratio=ratio,
            use_smote=(args.oversampler_kinase == "smote"),
            use_class_weight=args.use_class_weight_kinase,
            ch=args.stage2_ch, drop=args.stage2_drop, lr=args.stage2_lr,
            epochs=args.stage2_epochs, batch_size=args.stage2_batch,
            out_dir=outdir / "stage2_kinase"
        )
        fam2model[fam] = (mdl, le, kin_classes)

        # Save Stage-2 kinase expert weights and classes (for later prediction)
        fam_dir = outdir / "stage2_kinase" / safe_dirname(fam)
        fam_dir.mkdir(parents=True, exist_ok=True)
        mdl.save_weights(fam_dir / "kinase_expert.weights.h5")
        np.save(fam_dir / "kinase_classes.npy", np.array(kin_classes, dtype=object))

    #       Kinase    gated
    P_fam = tf.nn.softmax(logits_te, axis=1).numpy()
    Y_prob_kin, all_kin_cols = infer_kinase_gated(
        df_te, fam_names, fam2model, windows, args.backbone, cache_dir,
        P_fam=P_fam, topk=args.topk_family, soft_gate=args.soft_gate,
        esm_repr_mode=args.esm_repr_mode, esm_cache_dtype=args.esm_cache_dtype
    )

    # Global kinase evaluation (optional)
    if args.enable_global_kinase_eval:
        y_true_names = []
        mask_idx = []
        for i, (fam_true, kin_genes) in enumerate(zip(df_te["Family"].tolist(), df_te["KIN_GENE"].tolist())):
            items = SPLIT.split(kin_genes) if SPLIT.search(kin_genes) else [kin_genes]
            pick = None
            for kg in items:
                fam_of_k = normalize_family(fam_map.get(kg.strip().upper(), "Unknown"))
                if fam_of_k == fam_true and fam_of_k != "Unknown":
                    pick = f"{fam_true}::{kg.strip().upper()}"
                    break
            if pick and pick in all_kin_cols:
                y_true_names.append(pick)
                mask_idx.append(i)
        if len(mask_idx) >= 2:
            le_glob = LabelEncoder().fit(all_kin_cols)
            y_true_idx = le_glob.transform(y_true_names)
            y_prob_sub = Y_prob_kin[mask_idx][:, le_glob.transform(all_kin_cols)]
            eval_global_kinase(y_true_idx, y_prob_sub, all_kin_cols, outdir / "kinase_global")
        else:
            print("[WARN] Too few aligned test samples; skip global kinase metrics.")

    #    Family    Kinase K    #    Family    Kinase Kinase
    family_kin_summaries = []
    for fam in fam_names:
        fam_mask = (df_te["Family"] == fam).values
        if fam_mask.sum() < 2: continue
        mdl, le, kin_classes = fam2model[fam]
        #
        if args.backbone.lower() == "aa_onehot":
            embeds_sub = embed_aa(df_te[fam_mask], windows, None)
        else:
            embeds_sub = embed_esm(df_te[fam_mask], windows, cache_dir, args.backbone, repr_mode=args.esm_repr_mode,
                                   cache_dtype=args.esm_cache_dtype)
        Xs_sub = [embeds_sub[L] for L in windows]
        logits_sub = mdl.predict(Xs_sub, batch_size=256, verbose=0)
        prob_sub = tf.nn.softmax(logits_sub, axis=1).numpy()
        kin_names = [k for k in kin_classes if k != "[NOT_FAM]"]
        if len(kin_names) >= 2:
            mask_cols = np.array([k != "[NOT_FAM]" for k in kin_classes])
            prob_kin = prob_sub[:, mask_cols]
            fam_dir = outdir / "stage2_kinase" / safe_dirname(fam) / "test"
            fam_dir.mkdir(parents=True, exist_ok=True)
            metrics = eval_per_family_kinase_heatmap(df_te[fam_mask], prob_kin, kin_names, fam_dir, fam)
            if metrics is not None:
                family_kin_summaries.append(metrics)

    #      family   kinase-level
    if family_kin_summaries:
        summ_df = pd.DataFrame(family_kin_summaries)
        summ_dir = outdir / "stage2_kinase"
        summ_dir.mkdir(parents=True, exist_ok=True)
        summ_df.to_csv(summ_dir / "family_kinase_summary.csv", index=False)
        log(f"[Summary] Per-family kinase metrics saved to {summ_dir / 'family_kinase_summary.csv'}")

    #      CSV   v16
    topk = args.topk_family
    fam_topk_idx = np.argsort(-P_fam, axis=1)[:, :topk]
    fam_topk = [[fam_names[j] for j in fam_topk_idx[i]] for i in range(len(df_te))]
    kin_topk_idx = np.argsort(-Y_prob_kin, axis=1)[:, :min(5, Y_prob_kin.shape[1])]
    kin_topk = [[all_kin_cols[j] for j in kin_topk_idx[i]] for i in range(len(df_te))]
    export = pd.DataFrame({
        "ID": df_te["ID"].tolist(),
        "Family_true": df_te["Family"].tolist(),
        "Family_topk": [";".join(x) for x in fam_topk],
        "Kinase_top5": [";".join(x) for x in kin_topk],
    })
    export.to_csv(outdir / "user_view_test_topk.csv", index=False)

    log(f"Done. Outputs are in: {outdir}")
    # Dump run log to text file under outputs
    if args.save_run_log:
        log_path = outdir / "run_log.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                for line in RUN_LOG:
                    f.write(line + "\n")
            print(f"[LOG] Run log has been saved to: {log_path}")
        except Exception as e:
            print(f"[LOG] Failed to write run log: {e}")


if __name__ == "__main__":
    main()