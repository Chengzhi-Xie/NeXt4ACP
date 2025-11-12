# ACP_stack_ResNeXt_AAC_CKSAAP_PCP16_ablation.py
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

# First-layer base models (4ML)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Keras / TensorFlow for ResNeXt meta-learner
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

# ============== Features: AAC(20) + CKSAAP_k=0..5(2400) + PCP16(16) = 2436 dims ==============
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

# ============== Feature subset indices (for ablation combos) ==============
def feature_slices_2436():
    """Return slices within 2436 dims: AAC[0:20], CKSAAP[20:2420], PCP16[2420:2436]"""
    idx_aac = slice(0, 20)
    idx_ck  = slice(20, 2420)
    idx_pp  = slice(2420, 2436)
    return {"AAC": idx_aac, "CKSAAP": idx_ck, "PP16": idx_pp}

def select_features(X: np.ndarray, keys: List[str]) -> np.ndarray:
    sl = feature_slices_2436()
    cols = [X[:, sl[k]] for k in keys]
    return np.concatenate(cols, axis=1)

# ============== First-layer model set (four, fixed configuration) ==============
def get_base_models() -> Dict[str, object]:
    models = {}
    # LightGBM (prefer GPU; fallback to CPU)
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

    # XGBoost (prefer GPU; fallback to CPU)
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
        MCC=matthews_corrcoef(y_true, yb),
        Threshold=float(thr),
    )

# ============== ResNeXt meta-learner (fixed architecture) ==============
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

# ============== Single feature combo: full 4ML + ResNeXt pipeline ==============
def run_stack_for_feature_combo(
    X_train_sel: np.ndarray, X_test_sel: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
    folds, base_names: List[str], base_models: Dict[str, object], combo_name: str, random_state: int = 42
):
    print(f"\n========== [Combo] {combo_name} ==========")
    # 4 base models: produce OOF/Test probabilities with shared folds
    OOF_list, TEST_list = [], []
    for name in base_names:
        est = base_models[name]
        print(f"[Base] {name}: unified 5-fold OOF & holdout ...")
        oof, te = get_oof_and_holdout_with_folds(X_train_sel, y_train, X_test_sel, est, folds)
        OOF_list.append(oof)
        TEST_list.append(te)

    # Assemble meta features (column order consistent with base_names), then standardize
    meta_train = np.vstack(OOF_list).T.astype(np.float32)   # (N_train, 4)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)  # (N_test, 4)
    scaler = StandardScaler()
    META_TRAIN_S = scaler.fit_transform(meta_train)
    META_TEST_S  = scaler.transform(meta_test)

    # ResNeXt meta-learner (fixed)
    tf.keras.utils.set_random_seed(random_state)
    meta_model = build_resnext_meta(input_dim=META_TRAIN_S.shape[1])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    _ = meta_model.fit(META_TRAIN_S, y_train, epochs=70, batch_size=32, verbose=0,
                       validation_split=0.2, callbacks=[es, rl])

    # Probabilities & threshold
    p_tr = meta_model.predict(META_TRAIN_S, verbose=0).ravel()
    p_te = meta_model.predict(META_TEST_S,  verbose=0).ravel()
    thr = youden_threshold(y_train, p_tr)

    # Metrics
    m_train = {"Combo": combo_name, "Set": "Train(Apparent)", **metrics_from_proba(y_train, p_tr, thr)}
    m_test  = {"Combo": combo_name, "Set": "Test(Holdout)",   **metrics_from_proba(y_test,  p_te, thr)}

    # Cleanup
    del meta_model
    gc.collect()
    return (m_train, m_test), (meta_train, meta_test), thr

# ============== Main (batch ablation with 7 combos) ==============
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

    # ---- Read data & stratified split (1/6 as TEST) ----
    all_seq, all_y = read_fasta_pair_lines(fp_all)
    TEST_RATIO = 1.0 / 6.0
    RANDOM_STATE = 42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO, shuffle=True, stratify=all_y, random_state=RANDOM_STATE
    )
    print(f"[Loaded] total={len(all_seq)} | TRAIN={len(tr_seq)} | TEST={len(te_seq)}")

    # ---- Extract full 2436-dim features (once) ----
    print("[Feature] Extracting TRAIN ...")
    X_train_full = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test_full  = extract_features(te_seq,  k_max_cksaap=5)
    print(f"[Shapes] Train={X_train_full.shape}, Test={X_test_full.shape} (expect 2436 dims)")

    # ---- Unified 5-fold split ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train_full, y_train))

    base_models = get_base_models()
    base_names  = ["HistGradientBoosting", "LightGBM", "RandomForest", "XGBoost"]  # fixed order

    # ---- 7 feature combos (select by column slices) ----
    combos = [
        ("AAC", ["AAC"]),
        ("CKSAAP", ["CKSAAP"]),
        ("PP16", ["PP16"]),
        ("AAC+CKSAAP", ["AAC", "CKSAAP"]),
        ("CKSAAP+PP16", ["CKSAAP", "PP16"]),
        ("AAC+PP16", ["AAC", "PP16"]),
        ("AAC+CKSAAP+PP16", ["AAC", "CKSAAP", "PP16"]),
    ]

    rows = []
    thresholds = []
    meta_oofs = {}
    meta_tests = {}

    for cname, keys in combos:
        Xtr_sel = select_features(X_train_full, keys)
        Xte_sel = select_features(X_test_full,  keys)
        (m_tr, m_te), (meta_tr, meta_te), thr = run_stack_for_feature_combo(
            Xtr_sel, Xte_sel, y_train, y_test, folds, base_names, base_models, cname, random_state=RANDOM_STATE
        )
        rows.extend([m_tr, m_te])
        thresholds.append({"Combo": cname, "Threshold_from_Train": thr})
        meta_oofs[cname] = meta_tr
        meta_tests[cname] = meta_te

    # ---- Aggregate & save ----
    os.makedirs("stack_ablation_outputs", exist_ok=True)
    out_xlsx = os.path.join("stack_ablation_outputs",
                            f"ablation_4ML_ResNeXt_AAC_CKSAAP_PP16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    df_all = pd.DataFrame(rows)
    df_thr = pd.DataFrame(thresholds)

    # Ranking (Test set only)
    df_rank = (df_all[df_all["Set"].str.startswith("Test")]
               .sort_values(by="AUC", ascending=False)
               .reset_index(drop=True))

    print("\n=== Ablation Summary (sorted by Test AUC) ===")
    print(df_rank[["Combo", "AUC", "ACC", "Precision", "Recall", "F1", "MCC", "Threshold"]])

    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_all.to_excel(w, sheet_name="AllMetrics", index=False)
            df_thr.to_excel(w, sheet_name="Thresholds", index=False)
            df_rank.to_excel(w, sheet_name="TestAUC_Ranking", index=False)
            # Optional: save meta-features for each combo (for reproducibility/plotting)
            for cname in combos:
                name = cname[0]
                pd.DataFrame(meta_oofs[name], columns=base_names).to_excel(
                    w, sheet_name=f"{name}_MetaTrain_OOF", index=False
                )
                pd.DataFrame(meta_tests[name], columns=base_names).to_excel(
                    w, sheet_name=f"{name}_MetaTest_Preds", index=False
                )
        print(f"\nSaved to: {out_xlsx}")
    except Exception as e:
        print("[Warn] openpyxl not available, writing CSVs.", e)
        df_all.to_csv(out_xlsx.replace(".xlsx", "_AllMetrics.csv"), index=False)
        df_thr.to_csv(out_xlsx.replace(".xlsx", "_Thresholds.csv"), index=False)
        df_rank.to_csv(out_xlsx.replace(".xlsx", "_TestAUC_Ranking.csv"), index=False)

if __name__ == "__main__":
    np.random.seed(42)
    main()
