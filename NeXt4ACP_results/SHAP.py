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
                             f1_score, roc_auc_score, matthews_corrcoef, roc_curve)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import matplotlib.pyplot as plt

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

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
    "acidic": set("DE"), "aliphatic": set("ILV"), "aromatic": set("FYW"),
    "basic": set("KRH"), "charged": set("DEKRH"), "cyclic": set("P"),
    "hydrophilic": set("RNDQEHKST"), "hydrophobic": set("AILMFWV"),
    "hydroxylic": set("ST"), "neutral_pH": set("ACFGHILMNPQSTVWY"),
    "nonpolar": set("ACFGILMPVWY"), "polar": set("RNDQEHKTYS".replace(" ", "")),
    "small": set("ACDGNPSTV"), "large": set("EFHKLMQRWY"),
    "sulfur": set("CM"), "tiny": set("ACGST"),
}
PCP16_KEYS = list(PCP16_MAP.keys())

def _check_seq(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "")
    if not seq or any(ch not in AA_SET for ch in seq):
        raise ValueError(f"Invalid residues in sequence: {seq}")
    return seq

def aac_vector(sequence: str) -> np.ndarray:
    s = _check_seq(sequence)
    L = len(s); counts = {aa: 0.0 for aa in AA20}
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
            aac_vector(seq),                       # 20
            cksaap_vector(seq, k_max=k_max_cksaap),# 2400
            pcp16_vector(seq),                     # 16
        ], axis=0))
    X = np.vstack(feats).astype(np.float32)
    assert X.shape[1] == 2436, f"Unexpected feature dim {X.shape[1]} != 2436"
    return X

def build_feature_names(k_max_cksaap: int = 5) -> List[str]:
    names = []
    names.extend([f"AAC_{aa}" for aa in AA20])              # 20
    for dp in DIPEPTIDES: names.append(f"DPC_{dp}")         # gap0
    for k in range(1, k_max_cksaap + 1):                    # gaps 1..5
        for dp in DIPEPTIDES: names.append(f"CKSAAP_{dp}_gap{k}")
    names.extend([f"PCP16_{k}" for k in PCP16_KEYS])        # 16
    assert len(names) == 2436
    return names

# ============== First-layer model collection (four) ==============
def get_base_models() -> Dict[str, object]:
    models = {}
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
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )
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
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)
    return models

# ============== Generate OOF / holdout probabilities (shared folds) ==============
def get_oof_and_holdout_with_folds(X_tr, y_tr, X_ho, estimator, folds):
    oof = np.zeros(X_tr.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_ho.shape[0], len(folds)), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(folds):
        clf = clone(estimator)
        clf.fit(X_tr[tr_idx], y_tr[tr_idx])
        if hasattr(clf, "predict_proba"):
            oof[va_idx] = clf.predict_proba(X_tr[va_idx])[:, 1]
            hold_mat[:, k] = clf.predict_proba(X_ho)[:, 1]
        else:
            s_va = clf.decision_function(X_tr[va_idx])
            s_ho = clf.decision_function(X_ho)
            s_va = (s_va - s_va.min()) / (s_va.max() - s_va.min() + 1e-12)
            s_ho = (s_ho - s_ho.min()) / (s_ho.max() - s_ho.min() + 1e-12)
            oof[va_idx] = s_va; hold_mat[:, k] = s_ho
    return oof, hold_mat.mean(axis=1).astype(np.float32)

# ============== Evaluation & threshold ==============
def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr; idx = int(np.nanargmax(j))
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
        MCC=matthews_corrcoef(y_true, yb),
        Threshold=float(thr),
    )

# ============== ResNeXt meta-learner (lightweight) ==============
class ResNextBlock(layers.Layer):
    def __init__(self, units, cardinality=4, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        assert units % cardinality == 0
        self.units = units; self.cardinality = cardinality
        self.group_units = units // cardinality; self.dropout_rate = dropout_rate
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

# ============== SHAP interpretability: two-panel (A+B) plot with fallback logic ==============
def fit_lightgbm_for_shap() -> lgb.LGBMClassifier:
    try:
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=-1, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            n_jobs=-1, device="gpu"
        )
        _ = model.get_params()
    except Exception:
        try:
            model = lgb.LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                n_jobs=-1, device_type="gpu"
            )
        except Exception:
            model = lgb.LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
            )
    return model

def plot_shap_two_panel(trained_lgbm: lgb.LGBMClassifier,
                        X_ref: np.ndarray,
                        X_explain: np.ndarray,
                        feature_names: List[str],
                        out_dir: str) -> None:
    if not _SHAP_AVAILABLE:
        print("[SHAP] shap is not installed; skip plotting. Please `pip install shap` first.")
        return

    X_ref_df = pd.DataFrame(X_ref, columns=feature_names)
    X_exp_df = pd.DataFrame(X_explain, columns=feature_names)

    # Background set: subsample to avoid memory issues / leaf coverage problems
    bg = shap.sample(X_ref_df, min(2000, len(X_ref_df)))

    print("[SHAP] Computing TreeExplainer ...")
    try:
        explainer = shap.TreeExplainer(
            trained_lgbm, data=bg,
            feature_perturbation="tree_path_dependent"
        )
        shap_vals = explainer.shap_values(X_exp_df)
    except Exception as e1:
        print("[SHAP] tree_path_dependent failed, falling back to interventional:", e1)
        explainer = shap.TreeExplainer(
            trained_lgbm, data=bg,
            feature_perturbation="interventional"
        )
        shap_vals = explainer.shap_values(X_exp_df)

    # Binary classification return format compatibility
    shap_values = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9])

    ax0 = fig.add_subplot(gs[0, 0])
    plt.sca(ax0)
    shap.summary_plot(
        shap_values, X_exp_df,
        feature_names=feature_names,
        show=False, max_display=20
    )
    ax0.set_title("A", loc="left", fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    order = list(range(len(top_names)))[::-1]
    ax1.barh(range(len(top_names)), top_vals[order])
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels([top_names[i] for i in order])
    ax1.set_xlabel("Mean Absolute SHAP Value")
    ax1.set_title("B", loc="left", fontweight="bold")
    ax1.set_xlim(left=0)

    fig.tight_layout(w_pad=2.5)
    out_png = os.path.join(out_dir, "SHAP_Feature_Analysis.png")
    out_pdf = os.path.join(out_dir, "SHAP_Feature_Analysis.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Saved: {out_png}\n[SHAP] Saved: {out_pdf}")

# ============== Main ==============
def main():
    DATA_DIR = "data"
    fp_all = os.path.join(DATA_DIR, "antiCP2.txt")
    if not os.path.exists(fp_all):
        alt_all = os.path.join(os.getcwd(), "antiCP2.txt")
        if os.path.exists(alt_all): fp_all = alt_all
    if not os.path.exists(fp_all):
        raise FileNotFoundError(f"File not found in ./data or CWD: {fp_all}")

    all_seq, all_y = read_fasta_pair_lines(fp_all)
    TEST_RATIO = 1.0 / 6.0; RANDOM_STATE = 42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO, shuffle=True,
        stratify=all_y, random_state=RANDOM_STATE
    )
    print(f"[Loaded] total={len(all_seq)} | TRAIN={len(tr_seq)} | TEST={len(te_seq)}")

    print("[Feature] Extracting TRAIN ...")
    X_train = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test  = extract_features(te_seq,  k_max_cksaap=5)
    print(f"[Shapes] Train={X_train.shape}, Test={X_test.shape} (expect 2436 dims)")

    feature_names = build_feature_names(k_max_cksaap=5)

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

    meta_train = np.vstack(OOF_list).T.astype(np.float32)   # (N_train, 4)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)  # (N_test, 4)

    scaler = StandardScaler()
    META_TRAIN_S = scaler.fit_transform(meta_train)
    META_TEST_S  = scaler.transform(meta_test)

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    meta_model = build_resnext_meta(input_dim=META_TRAIN_S.shape[1])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    meta_model.fit(META_TRAIN_S, y_train, epochs=70, batch_size=32, verbose=1,
                   validation_split=0.2, callbacks=[es, rl])

    p_tr = meta_model.predict(META_TRAIN_S, verbose=0).ravel()
    p_te = meta_model.predict(META_TEST_S,  verbose=0).ravel()

    thr = youden_threshold(y_train, p_tr)

    m_train = {"Set": "Train(Apparent)", **metrics_from_proba(y_train, p_tr, thr)}
    m_test  = {"Set": "Test(Holdout)",   **metrics_from_proba(y_test,  p_te, thr)}
    df_meta = pd.DataFrame([m_train, m_test])

    os.makedirs("stack_holdout_outputs", exist_ok=True)
    out_xlsx = os.path.join("stack_holdout_outputs",
                            f"stack_ResNeXt_AAC_CKSAAP_PCP16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
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

    # ---- SHAP: LightGBM explanation on 2436-dim raw features ----
    try:
        lgbm_for_shap = fit_lightgbm_for_shap()
        lgbm_for_shap.fit(X_train, y_train)
        plot_shap_two_panel(
            trained_lgbm=lgbm_for_shap,
            X_ref=X_train, X_explain=X_test,
            feature_names=feature_names,
            out_dir="stack_holdout_outputs"
        )
    except Exception as e:
        print("[SHAP] Computation or plotting failed:", e)

    del meta_model
    gc.collect()

if __name__ == "__main__":
    np.random.seed(42)
    main()
