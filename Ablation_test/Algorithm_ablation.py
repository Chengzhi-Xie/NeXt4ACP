# ACP_algo_ablation_4ML_ResNeXt_AAC_CKSAAP_PP16.py
# -*- coding: utf-8 -*-
import os, warnings, gc
from datetime import datetime
from itertools import product, combinations
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef, roc_curve)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

# First-layer four models
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

# ============== Data reading (FASTA two-line: odd header, even sequence) ==============
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

# ============== Features: AAC(20) + CKSAAP (k=0..5 → 2400) + PCP16(16) = 2436 dims ==============
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
    "polar": set("RNDQEHKTYS"),
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
    return np.array(out, dtype=np.float32)  # 2400

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

# ============== Four models ==============
def get_base_models() -> Dict[str, object]:
    models = {}
    # LightGBM (per-fold early stopping)
    try:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=2000, max_depth=-1, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            n_jobs=-1, device="gpu"
        )
        _ = models["LightGBM"].get_params()
    except Exception:
        try:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=2000, max_depth=-1, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                n_jobs=-1, device_type="gpu"
            )
        except Exception:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=2000, max_depth=-1, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
            )
    # RandomForest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=42
    )
    # XGBoost (per-fold early stopping)
    try:
        models["XGBoost"] = XGBClassifier(
            n_estimators=2000, max_depth=6, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
            random_state=42, n_jobs=-1, tree_method="gpu_hist", predictor="gpu_predictor"
        )
    except Exception:
        models["XGBoost"] = XGBClassifier(
            n_estimators=2000, max_depth=6, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
            random_state=42, n_jobs=-1, tree_method="hist"
        )
    # HistGradientBoosting
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(random_state=42)
    return models

# ============== Single-fold fit (with early stopping) and safe prediction ==============
def _predict_proba_safely(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return s

def _fit_one_fold_with_es(base_est, X_tr, y_tr, X_va, y_va, X_ho):
    est = clone(base_est)
    name = est.__class__.__name__
    # LightGBM early stopping
    if name == "LGBMClassifier":
        try:
            est.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
                verbose=False
            )
            return est.predict_proba(X_va)[:, 1], est.predict_proba(X_ho)[:, 1]
        except Exception:
            pass
    # XGBoost early stopping
    if name == "XGBClassifier":
        try:
            est.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
                early_stopping_rounds=100
            )
            best_it = getattr(est, "best_iteration", None)
            if best_it is not None:
                try:
                    p_va = est.predict_proba(X_va, iteration_range=(0, best_it + 1))[:, 1]
                    p_ho = est.predict_proba(X_ho, iteration_range=(0, best_it + 1))[:, 1]
                except TypeError:
                    p_va = est.predict_proba(X_va, ntree_limit=best_it + 1)[:, 1]
                    p_ho = est.predict_proba(X_ho, ntree_limit=best_it + 1)[:, 1]
            else:
                p_va = est.predict_proba(X_va)[:, 1]
                p_ho = est.predict_proba(X_ho)[:, 1]
            return p_va, p_ho
        except Exception:
            pass
    # Other models: standard fit
    est.fit(X_tr, y_tr)
    return _predict_proba_safely(est, X_va), _predict_proba_safely(est, X_ho)

# ============== Generate OOF / holdout probabilities (shared folds + early stopping) ==============
def get_oof_and_holdout_with_folds(X_tr, y_tr, X_ho, estimator, folds):
    oof = np.zeros(X_tr.shape[0], dtype=np.float32)
    hold_mat = np.zeros((X_ho.shape[0], len(folds)), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
        ytr, yva = y_tr[tr_idx], y_tr[va_idx]
        p_va, p_ho = _fit_one_fold_with_es(estimator, Xtr, ytr, Xva, yva, X_ho)
        oof[va_idx] = p_va
        hold_mat[:, k] = p_ho
    return oof, hold_mat.mean(axis=1).astype(np.float32)

# ============== Threshold & metrics ==============
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

# ============== ResNeXt (meta-learner & raw-feature model) ==============
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

def build_resnext_raw(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = ResNextBlock(128, cardinality=8, dropout_rate=0.3)(x)
    x = ResNextBlock(128, cardinality=8, dropout_rate=0.3)(x)
    x = ResNextBlock(128, cardinality=8, dropout_rate=0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    opt = optimizers.Adam(learning_rate=3e-4)
    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return m

def fit_resnext_ensemble(X_tr, y_tr, X_te, build_fn, seeds=(42, 2024, 7, 99, 123)):
    preds_tr, preds_te = [], []
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    for sd in seeds:
        tf.keras.utils.set_random_seed(sd)
        m = build_fn(X_tr.shape[1])
        m.fit(X_tr, y_tr, epochs=70, batch_size=32, verbose=0,
              validation_split=0.2, callbacks=[es, rl])
        preds_tr.append(m.predict(X_tr, verbose=0).ravel())
        preds_te.append(m.predict(X_te,  verbose=0).ravel())
        del m
        gc.collect()
    return np.mean(preds_tr, axis=0), np.mean(preds_te, axis=0)

# ============== OOF/Test probabilities for 4ML (computed once) ==============
def compute_meta_predictions_all4(X_train, y_train, X_test, folds, base_models, base_names):
    OOF_list, TEST_list = [], []
    for name in base_names:
        est = base_models[name]
        print(f"[Base] {name}: 5-fold with early-stopping ...")
        oof, te = get_oof_and_holdout_with_folds(X_train, y_train, X_test, est, folds)
        OOF_list.append(oof)
        TEST_list.append(te)
    meta_train = np.vstack(OOF_list).T.astype(np.float32)   # (N_train, 4)
    meta_test  = np.vstack(TEST_list).T.astype(np.float32)  # (N_test, 4)
    return meta_train, meta_test

# ============== Evaluate ablation variants ==============
def eval_resnext_on_meta_subset(meta_train, meta_test, y_train, y_test, subset_idxs, label):
    # Standardize
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(meta_train[:, subset_idxs])
    Xte = scaler.transform(meta_test[:, subset_idxs])
    # Multi-seed ResNeXt (meta-learner)
    p_tr, p_te = fit_resnext_ensemble(Xtr, y_train, Xte, build_resnext_meta)
    thr = youden_threshold(y_train, p_tr)
    m_tr = {"Variant": label, "Set": "Train(Apparent)", **metrics_from_proba(y_train, p_tr, thr)}
    m_te = {"Variant": label, "Set": "Test(Holdout)",   **metrics_from_proba(y_test,  p_te, thr)}
    return m_tr, m_te, thr

def eval_softvote_on_meta_subset(meta_train, meta_test, y_train, y_test, subset_idxs, label):
    p_tr = meta_train[:, subset_idxs].mean(axis=1)
    p_te = meta_test[:, subset_idxs].mean(axis=1)
    thr = youden_threshold(y_train, p_tr)
    m_tr = {"Variant": label, "Set": "Train(Apparent)", **metrics_from_proba(y_train, p_tr, thr)}
    m_te = {"Variant": label, "Set": "Test(Holdout)",   **metrics_from_proba(y_test,  p_te, thr)}
    return m_tr, m_te, thr

def eval_resnext_on_raw_features(X_train_full, X_test_full, y_train, y_test, label="ResNeXt (raw 2436)"):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_full)
    Xte = scaler.transform(X_test_full)
    p_tr, p_te = fit_resnext_ensemble(Xtr, y_train, Xte, build_resnext_raw)
    thr = youden_threshold(y_train, p_tr)
    m_tr = {"Variant": label, "Set": "Train(Apparent)", **metrics_from_proba(y_train, p_tr, thr)}
    m_te = {"Variant": label, "Set": "Test(Holdout)",   **metrics_from_proba(y_test,  p_te, thr)}
    return m_tr, m_te, thr

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

    # ---- Read data & stratified split (1/6 as TEST) ----
    all_seq, all_y = read_fasta_pair_lines(fp_all)
    TEST_RATIO = 1.0 / 6.0
    RANDOM_STATE = 42
    tr_seq, te_seq, y_train, y_test = train_test_split(
        all_seq, all_y, test_size=TEST_RATIO, shuffle=True, stratify=all_y, random_state=RANDOM_STATE
    )
    print(f"[Loaded] total={len(all_seq)} | TRAIN={len(tr_seq)} | TEST={len(te_seq)}")

    # ---- Extract features (2436 dims, fixed AAC+CKSAAP+PP16) ----
    print("[Feature] Extracting TRAIN ...")
    X_train_full = extract_features(tr_seq, k_max_cksaap=5)
    print("[Feature] Extracting TEST  ...")
    X_test_full  = extract_features(te_seq,  k_max_cksaap=5)
    print(f"[Shapes] Train={X_train_full.shape}, Test={X_test_full.shape} (expect 2436 dims)")

    # ---- Unified 5 folds, compute OOF/holdout probabilities for four models ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train_full, y_train))
    base_models = get_base_models()
    base_names  = ["HistGradientBoosting", "LightGBM", "RandomForest", "XGBoost"]
    name_to_idx = {n: i for i, n in enumerate(base_names)}
    meta_train, meta_test = compute_meta_predictions_all4(
        X_train_full, y_train, X_test_full, folds, base_models, base_names
    )

    rows, thrs = [], []

    # ---- 4ML + ResNeXt (baseline) ----
    idx_all = np.arange(4)
    mtr, mte, thr = eval_resnext_on_meta_subset(meta_train, meta_test, y_train, y_test, idx_all,
                                                "4ML+ResNeXt (HGB+LGB+RF+XGB)")
    rows.extend([mtr, mte]); thrs.append({"Variant": mtr["Variant"], "Threshold_from_Train": thr})

    # ---- 4ML (soft vote) ----
    mtr, mte, thr = eval_softvote_on_meta_subset(meta_train, meta_test, y_train, y_test, idx_all,
                                                 "4ML (soft-vote: HGB+LGB+RF+XGB)")
    rows.extend([mtr, mte]); thrs.append({"Variant": mtr["Variant"], "Threshold_from_Train": thr})

    # ---- ResNeXt (directly using 2436-dim raw features) ----
    mtr, mte, thr = eval_resnext_on_raw_features(X_train_full, X_test_full, y_train, y_test,
                                                 "ResNeXt (raw 2436)")
    rows.extend([mtr, mte]); thrs.append({"Variant": mtr["Variant"], "Threshold_from_Train": thr})

    # ---- 3ML + ResNeXt (choose 3 of 4, 4 variants) ----
    for comb in combinations(range(4), 3):
        label = "3ML+ResNeXt (" + "+".join([base_names[i].replace("HistGradientBoosting","HGB").replace("LightGBM","LGB").replace("RandomForest","RF").replace("XGBoost","XGB") for i in comb]) + ")"
        mtr, mte, thr = eval_resnext_on_meta_subset(meta_train, meta_test, y_train, y_test, np.array(comb), label)
        rows.extend([mtr, mte]); thrs.append({"Variant": mtr["Variant"], "Threshold_from_Train": thr})

    # ---- 2ML + ResNeXt (choose 2 of 4, 6 variants) ----
    for comb in combinations(range(4), 2):
        label = "2ML+ResNeXt (" + "+".join([base_names[i].replace("HistGradientBoosting","HGB").replace("LightGBM","LGB").replace("RandomForest","RF").replace("XGBoost","XGB") for i in comb]) + ")"
        mtr, mte, thr = eval_resnext_on_meta_subset(meta_train, meta_test, y_train, y_test, np.array(comb), label)
        rows.extend([mtr, mte]); thrs.append({"Variant": mtr["Variant"], "Threshold_from_Train": thr})

    # ---- 1ML + ResNeXt (choose 1 of 4, 4 variants) ----
    for i in range(4):
        label = "1ML+ResNeXt (" + base_names[i].replace("HistGradientBoosting","HGB").replace("LightGBM","LGB").replace("RandomForest","RF").replace("XGBoost","XGB") + ")"
        mtr, mte, thr = eval_resnext_on_meta_subset(meta_train, meta_test, y_train, y_test, np.array([i]), label)
        rows.extend([mtr, mte]); thrs.append({"Variant": mtr["Variant"], "Threshold_from_Train": thr})

    # ---- Aggregate, rank, and save ----
    df_all = pd.DataFrame(rows)
    df_thr = pd.DataFrame(thrs)
    df_rank = (df_all[df_all["Set"].str.startswith("Test")]
               .sort_values(by="AUC", ascending=False)
               .reset_index(drop=True))

    print("\n=== Algorithm Ablation Summary (sorted by Test AUC) ===")
    print(df_rank[["Variant", "AUC", "ACC", "Precision", "Recall", "F1", "MCC", "Threshold"]])

    os.makedirs("algo_ablation_outputs", exist_ok=True)
    out_xlsx = os.path.join("algo_ablation_outputs",
                            f"algo_ablation_4ML_ResNeXt_AAC_CKSAAP_PP16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df_all.to_excel(w, sheet_name="AllMetrics", index=False)
            df_thr.to_excel(w, sheet_name="Thresholds", index=False)
            df_rank.to_excel(w, sheet_name="TestAUC_Ranking", index=False)
            # Also save meta OOF/Test for the four base models (for reproducibility/plotting)
            base_cols = ["HGB","LGB","RF","XGB"]
            pd.DataFrame(meta_train, columns=base_cols).to_excel(w, sheet_name="MetaTrain_OOF_all4", index=False)
            pd.DataFrame(meta_test,  columns=base_cols).to_excel(w, sheet_name="MetaTest_Preds_all4", index=False)
        print(f"\nSaved to: {out_xlsx}")
    except Exception as e:
        print("[Warn] openpyxl not available, writing CSVs.", e)
        df_all.to_csv(out_xlsx.replace(".xlsx", "_AllMetrics.csv"), index=False)
        df_thr.to_csv(out_xlsx.replace(".xlsx", "_Thresholds.csv"), index=False)
        df_rank.to_csv(out_xlsx.replace(".xlsx", "_TestAUC_Ranking.csv"), index=False)

if __name__ == "__main__":
    np.random.seed(42)
    main()
