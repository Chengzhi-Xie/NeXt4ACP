# ACP_stack_ResNeXt_AAC_CKSAAP_PCP16.py
# -*- coding: utf-8 -*-
import os, warnings, gc
from datetime import datetime
from itertools import product
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)  # Confusion matrix
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # PCA

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Monochrome gradient

# Four first-layer models
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Keras / TensorFlow for ResNeXt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# ============== TF GPU (on-demand memory growth) ==============
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[TF] Using {len(gpus)} GPU(s) with memory growth.")
    except Exception as e:
        print("[TF] set_memory_growth failed:", e)
else:
    print("[TF] No GPU found, running on CPU.")

# ============== Data reading (FASTA two-line: odd lines are headers, even lines are sequences) ==============
def read_fasta_pair_lines(file_path: str) -> Tuple[List[str], np.ndarray]:
    sequences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for i in range(0, len(lines), 2):
        header = lines[i]
        seq = lines[i + 1].strip().upper().replace(" ", "")
        label = 1 if "positive" in header.lower() else 0
        sequences.append(seq)
        labels.append(label)
    return sequences, np.array(labels, dtype=np.int32)

# ============== Features: AAC + CKSAAP_k + PCP16 (total 2436 dims) ==============
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA20)
DIPEPTIDES = [''.join(p) for p in product(AA20, repeat=2)]  # 400

PCP16_MAP = {
    "acidic": set("DE"),
    "aliphatic": set("ILV"),
    "aromatic": set("FYW"),
    "basic": set("KRH"),
    "charged": set("DEKRH"),
    "cyclic": set("P"),
    "hydrophilic": set("RNDQEHKST"),
    "hydrophobic": set("AILMFWV"),
    "hydroxylic": set("ST"),
    "neutral_pH": set("ACFGHILMNPQSTVWY"),
    "nonpolar": set("ACFGILMPVWY"),
    "polar": set("RNDQEHKTYS".replace(" ", "")),
    "small": set("ACDGNPSTV"),
    "large": set("EFHKLMQRWY"),
    "sulfur": set("CM"),
    "tiny": set("ACGST"),
}
PCP16_KEYS = list(PCP16_MAP.keys())

def _check_seq(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "")
    if not seq or any(ch not in AA_SET for ch in seq):
        raise ValueError(f"Invalid residues in sequence: {seq}")
    return seq

def aac_vector(sequence: str) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s)
    counts = {aa: 0.0 for aa in AA20}
    for ch in s: counts[ch] += 1.0
    if L > 0:
        for aa in counts: counts[aa] /= L
    return np.array([counts[aa] for aa in AA20], dtype=np.float32)  # 20

def cksaap_vector(sequence: str, k_max: int = 5) -> np.ndarray:
    s = _check_seq(sequence); L = len(s); out = []
    for k in range(k_max + 1):
        counts = dict.fromkeys(DIPEPTIDES, 0.0)
        if L >= k + 2:
            for i in range(L - 1 - k):
                pair = s[i] + s[i + 1 + k]; counts[pair] += 1.0
            total = sum(counts.values())
            if total > 0:
                for dp in counts: counts[dp] /= total
        out.extend([counts[dp] for dp in DIPEPTIDES])
    return np.array(out, dtype=np.float32)  # (k_max+1)*400 = 2400

def pcp16_vector(sequence: str) -> np.ndarray:
    s = _check_seq(sequence); L = len(s)
    return np.array([sum(ch in PCP16_MAP[key] for ch in s) / L for key in PCP16_KEYS], dtype=np.float32)  # 16

def extract_features(seqs: List[str], k_max_cksaap: int = 5) -> np.ndarray:
    feats = []
    for seq in seqs:
        feats.append(np.concatenate([
            aac_vector(seq),                        # 20
            cksaap_vector(seq, k_max=k_max_cksaap), # 2400
            pcp16_vector(seq),                      # 16
        ], axis=0))
    X = np.vstack(feats).astype(np.float32)
    assert X.shape[1] == 2436, f"Unexpected feature dim {X.shape[1]} != 2436"
    return X

# ============== First-layer model collection (four) ==============
def get_base_models() -> Dict[str, object]:
    models = {}
    # LightGBM (prefer GPU, fall back to CPU)
    try:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=500, max_depth=-1, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            n_jobs=-1, device="gpu"
        )
        _ = models["LightGBM"].get_params()
    except Exception:
        try:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                n_jobs=-1, device_type="gpu"
            )
        except Exception:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
            )

    # RandomForest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )

    # XGBoost (prefer GPU, fall back to CPU)
    try:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="gpu_hist", predictor="gpu_predictor"
        )
    except Exception:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="hist"
        )

    # HistGradientBoosting
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)
    return models

# ============== Generate OOF / holdout probabilities (shared folds) ==============
def get_oof_and_holdout_with_folds(X_tr, y_tr, X_ho, estimator, folds):
    """
    Returns:
      oof:  (N_train,)  — OOF probabilities for this model
      hold: (N_holdout,) — predict holdout with each fold model, then average across folds
    """
    oof = np.zeros(X_tr.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_ho.shape[0], len(folds)), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(folds):
        clf = clone(estimator)
        clf.fit(X_tr[tr_idx], y_tr[tr_idx])
        if hasattr(clf, "predict_proba"):
            oof[va_idx] = clf.predict_proba(X_tr[va_idx])[:, 1]
            hold_mat[:, k] = clf.predict_proba(X_ho)[:, 1]
        else:
            # Fallback: linearly scale decision_function
            s_va = clf.decision_function(X_tr[va_idx])
            s_ho = clf.decision_function(X_ho)
            s_va = (s_va - s_va.min()) / (s_va.max() - s_va.min() + 1e-12)
            s_ho = (s_ho - s_ho.min()) / (s_ho.max() - s_ho.min() + 1e-12)
            oof[va_idx] = s_va
            hold_mat[:, k] = s_ho
    return oof, hold_mat.mean(axis=1).astype(np.float32)

# ============== Evaluation & threshold ==============
def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    best = float(thr[idx]) if np.isfinite(thr[idx]) else 0.5
    return float(np.clip(best, 0.0, 1.0))

def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    yb = (proba >= thr).astype(int)
    return dict(
        ACC=accuracy_score(y_true, yb),
        Precision=precision_score(y_true, yb, zero_division=0),
        Recall=recall_score(y_true, yb),
        F1=f1_score(y_true, yb),
        AUC=roc_auc_score(y_true, proba),
        AP=average_precision_score(y_true, proba),
        MCC=matthews_corrcoef(y_true, yb),
        Threshold=float(thr),
    )

# ============== ROC / PRC plotting utilities ==============
def _save_curve_points_roc(fpr, tpr, thr, save_csv):
    df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr})
    df.to_csv(save_csv, index=False)

def _save_curve_points_prc(precision, recall, thr, save_csv):
    thr_full = np.concatenate([thr, [np.nan]]) if thr is not None else np.full_like(precision, np.nan)
    df = pd.DataFrame({"Recall": recall, "Precision": precision, "Threshold": thr_full})
    df.to_csv(save_csv, index=False)

def plot_roc_prc(y_true, proba, set_name: str, out_dir: str,
                 title_prefix: str = "ResNeXt Stacking (AAC+CKSAAP+PCP16)",
                 title_override: Tuple[str, str] | None = None):
    """
    title_override: (roc_title, prc_title) — if provided, fully override the title texts
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ROC
    fpr, tpr, thr = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold', color='black')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold', color='black')
    roc_title = title_override[0] if title_override else f'{title_prefix} — {set_name} ROC'
    plt.title(roc_title, fontsize=14, fontweight='bold', color='black')
    plt.legend(loc="lower right")
    roc_png = os.path.join(out_dir, f"{ts}_{set_name}_ROC.png")
    plt.tight_layout(); plt.savefig(roc_png, dpi=300); plt.close()
    _save_curve_points_roc(fpr, tpr, thr, os.path.join(out_dir, f"{ts}_{set_name}_ROC_points.csv"))

    # PRC
    precision, recall, thr_pr = precision_recall_curve(y_true, proba)
    ap_val = average_precision_score(y_true, proba)
    plt.figure()
    plt.plot(recall, precision, label=f"PRC (AP = {ap_val:.4f})", linewidth=2)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold', color='black')
    plt.ylabel('Precision', fontsize=12, fontweight='bold', color='black')
    prc_title = title_override[1] if title_override else f'{title_prefix} — {set_name} PRC'
    plt.title(prc_title, fontsize=14, fontweight='bold', color='black')
    plt.legend(loc="lower left")
    prc_png = os.path.join(out_dir, f"{ts}_{set_name}_PRC.png")
    plt.tight_layout(); plt.savefig(prc_png, dpi=300); plt.close()
    _save_curve_points_prc(precision, recall, thr_pr, os.path.join(out_dir, f"{ts}_{set_name}_PRC_points.csv"))

    print(f"[Plot] {set_name}: ROC→{roc_png} | PRC→{prc_png}")

# ============== PCA plotting utility ==============
def _scatter_pca(X2d, y, title, save_png):
    plt.figure(figsize=(6, 5))
    y = np.asarray(y).astype(int)
    plt.scatter(X2d[y == 1, 0], X2d[y == 1, 1], label='Positive', s=18)
    plt.scatter(X2d[y == 0, 0], X2d[y == 0, 1], label='Negative', s=18)
    plt.xlabel('PC1', fontsize=12, fontweight='bold', color='black')
    plt.ylabel('PC2', fontsize=12, fontweight='bold', color='black')
    plt.title(title, fontsize=14, fontweight='bold', color='black')
    plt.xticks(fontsize=10, fontweight='bold', color='black')
    plt.yticks(fontsize=10, fontweight='bold', color='black')
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(save_png, dpi=300)
    plt.close()
    print(f"[PCA] Saved: {save_png}")

# ============== Confusion matrix plotting utility ==============
def _single_color_cmap(hex_color: str):
    return LinearSegmentedColormap.from_list('single_color', ['#FFFFFF', hex_color], N=256)

def plot_confmat(cm: np.ndarray, title: str, out_png: str, main_color: str = "#d32f2f"):
    cmap = _single_color_cmap(main_color)
    plt.figure(figsize=(5.2, 4.6))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold', color='black')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = ['Negative', 'Positive']
    plt.xticks(np.arange(2), ticks, fontsize=11, fontweight='bold', color='black')
    plt.yticks(np.arange(2), ticks, fontsize=11, fontweight='bold', color='black')
    plt.ylabel('True label', fontsize=12, fontweight='bold', color='black')
    plt.xlabel('Predicted label', fontsize=12, fontweight='bold', color='black')

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            plt.text(j, i, f"{val}", ha='center', va='center',
                     color='white' if val > thresh else 'black',
                     fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[CM] Saved: {out_png}")

# ============== ResNeXt meta-learner (lightweight) ==============
class ResNextBlock(layers.Layer):
    def __init__(self, units, cardinality=4, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        assert units % cardinality == 0, "units must be divisible by cardinality"
        self.units = units
        self.cardinality = cardinality
        self.group_units = units // cardinality
        self.dropout_rate = dropout_rate
        self.branches = []
        for _ in range(cardinality):
            branch = models.Sequential([
                layers.Dense(self.group_units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                layers.Dense(self.group_units)
            ])
            self.branches.append(branch)
        self.skip_dense = layers.Dense(units, use_bias=False)
        self.activation = layers.Activation('relu')

    def call(self, inputs, training=False):
        splits = tf.split(inputs, self.cardinality, axis=-1)
        branch_outputs = [branch(splits[i], training=training) for i, branch in enumerate(self.branches)]
        aggregated = layers.concatenate(branch_outputs, axis=-1)
        skip = self.skip_dense(inputs)
        return self.activation(aggregated + skip)

def build_resnext_meta(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    opt = optimizers.Adam(learning_rate=3e-4)
    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ============== Main ==============
def main():
    # ---- Fixed/fallback path: antiCP2.txt ----
    DATA_DIR = "data"
    fp_all = os.path.join(DATA_DIR, "antiCP2.txt")
    if not os.path.exists(fp_all):
        alt_all = os.path.join(os.getcwd(), "antiCP2.txt")
        if os.path.exists(alt_all):
            fp_all = alt_all
    if not os.path.exists(fp_all):
        raise FileNotFoundError(f"File not found in ./data or CWD: {fp_all}")

    # ---- Read data & stratified split (1/6 as TEST)----
    all_seq, all_y = read_fasta_pair_lines(fp_all)
    TEST_RATIO = 1.0 / 6.0
    RANDOM_STATE = 42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO, shuffle=True, stratify=all_y, random_state=RANDOM_STATE
    )
    print(f"[Loaded] total={len(all_seq)} | TRAIN={len(tr_seq)} | TEST={len(te_seq)}")

    # ---- Extract features (2436 dims)----
    print("[Feature] Extracting TRAIN ...")
    X_train = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test  = extract_features(te_seq,  k_max_cksaap=5)
    print(f"[Shapes] Train={X_train.shape}, Test={X_test.shape} (expect 2436 dims)")

    # ---- Use unified 5 folds to generate OOF / holdout probabilities for four base models ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train, y_train))

    base_models = get_base_models()
    base_names  = ["HistGradientBoosting", "LightGBM", "RandomForest", "XGBoost"]
    models_in_use = {name: base_models[name] for name in base_names}

    OOF_list, TEST_list = [], []
    for name in base_names:
        est = models_in_use[name]
        print(f"\n[Base] {name}: unified 5-fold OOF & holdout ...")
        oof, te = get_oof_and_holdout_with_folds(X_train, y_train, X_test, est, folds)
        OOF_list.append(oof); TEST_list.append(te)

    # ---- Assemble meta features (columns aligned with base_names)----
    meta_train = np.vstack(OOF_list).T.astype(np.float32)   # (N_train, 4)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)  # (N_test, 4)

    # ---- Standardization (fit on train, transform test)----
    scaler = StandardScaler()
    META_TRAIN_S = scaler.fit_transform(meta_train)
    META_TEST_S  = scaler.transform(meta_test)

    # ---- Train ResNeXt (second layer)----
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    meta_model = build_resnext_meta(input_dim=META_TRAIN_S.shape[1])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    hist = meta_model.fit(META_TRAIN_S, y_train, epochs=70, batch_size=32, verbose=1,
                          validation_split=0.2, callbacks=[es, rl])

    # ---- Probabilities on TRAIN (apparent) and TEST (holdout)----
    p_tr = meta_model.predict(META_TRAIN_S, verbose=0).ravel()
    p_te = meta_model.predict(META_TEST_S,  verbose=0).ravel()

    # ---- Threshold: choose by Youden's J using only training probabilities (apparent) ----
    thr = youden_threshold(y_train, p_tr)

    # ---- Compute metrics ----
    m_train = {"Set": "Train(Apparent)", **metrics_from_proba(y_train, p_tr, thr)}
    m_test  = {"Set": "Test(Holdout)",   **metrics_from_proba(y_test,  p_te, thr)}
    df_meta = pd.DataFrame([m_train, m_test])

    # ---- Output directory ----
    out_dir = "stack_holdout_outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot and save ROC/PRC
    # Train: keep the default title style
    plot_roc_prc(y_train, p_tr, "Train", out_dir)
    # Test: override titles with your specified text
    plot_roc_prc(y_test,  p_te, "Test",  out_dir,
                 title_override=("NeXt4ACP-Test ROC", "NeXt4ACP-Test PRC"))

    # ================= Confusion matrices (Train: red; Test: deep blue) =================
    yb_train = (p_tr >= thr).astype(int)
    yb_test  = (p_te >= thr).astype(int)
    cm_train = confusion_matrix(y_train, yb_train, labels=[0, 1])
    cm_test  = confusion_matrix(y_test,  yb_test,  labels=[0, 1])

    plot_confmat(cm_train,
                 title="Confusion Matrix — Train",
                 out_png=os.path.join(out_dir, f"{ts}_CM_Train_red.png"),
                 main_color="#d32f2f")  # Red
    plot_confmat(cm_test,
                 title="Confusion Matrix — Test",
                 out_png=os.path.join(out_dir, f"{ts}_CM_Test_deepblue.png"),
                 main_color="#0b3d91")  # Deep blue

    # ================= PCA (3 plots) =================
    # (1) Raw features PCA — fit on TRAIN (avoid leakage), plot Train/Test
    scaler_raw = StandardScaler().fit(X_train)
    X_train_s = scaler_raw.transform(X_train)
    X_test_s  = scaler_raw.transform(X_test)
    pca_raw = PCA(n_components=2).fit(X_train_s)

    raw_train_2d = pca_raw.transform(X_train_s)
    raw_test_2d  = pca_raw.transform(X_test_s)

    _scatter_pca(raw_train_2d, y_train,
                 title="Raw Features PCA — Train",
                 save_png=os.path.join(out_dir, f"{ts}_PCA_Raw_Train.png"))
    _scatter_pca(raw_test_2d, y_test,
                 title="Raw Features PCA — Test",
                 save_png=os.path.join(out_dir, f"{ts}_PCA_Raw_Test.png"))

    # (3) Learned meta-representation PCA — fit on TRAIN representation, plot Test
    intermediate = tf.keras.Model(inputs=meta_model.input,
                                  outputs=meta_model.layers[-2].output)
    rep_train = intermediate.predict(META_TRAIN_S, verbose=0)
    rep_test  = intermediate.predict(META_TEST_S,  verbose=0)

    scaler_rep = StandardScaler().fit(rep_train)
    rep_test_s = scaler_rep.transform(rep_test)
    pca_rep = PCA(n_components=2).fit(scaler_rep.transform(rep_train))
    rep_test_2d = pca_rep.transform(rep_test_s)

    _scatter_pca(rep_test_2d, y_test,
                 title="Learned Representation PCA — Test",
                 save_png=os.path.join(out_dir, f"{ts}_PCA_LearnedRep_Test.png"))

    # ---- Save Excel (including AP metric and threshold, etc.) ----
    out_xlsx = os.path.join(out_dir, f"stack_ResNeXt_AAC_CKSAAP_PCP16_{ts}.xlsx")
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_meta.to_excel(w, sheet_name="Meta_Train_vs_Test", index=False)
            pd.DataFrame(meta_train, columns=base_names).to_excel(w, sheet_name="MetaTrain_OOF", index=False)
            pd.DataFrame(meta_test,  columns=base_names).to_excel(w, sheet_name="MetaTest_Preds", index=False)
            pd.DataFrame([{"Threshold_from_Train": thr}]).to_excel(w, sheet_name="Meta_Threshold", index=False)
        print(f"\nSaved to: {out_xlsx}")
    except Exception as e:
        print("[Warn] openpyxl not available, writing CSVs.", e)
        df_meta.to_csv(out_xlsx.replace(".xlsx", "_meta.csv"), index=False)

    print("\n=== ResNeXt Stacking (AAC+CKSAAP+PCP16) ===")
    print(df_meta)

    # Release
    del meta_model
    gc.collect()

if __name__ == "__main__":
    np.random.seed(42)
    main()
