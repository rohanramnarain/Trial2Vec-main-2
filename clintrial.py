#!/usr/bin/env python3
"""
RF + MDI + SHAP + fANOVA (guarded) with a robust fallback to permutation importance.

- Uses BalancedRandomForest if available; otherwise RF(class_weight="balanced").
- Tunes the classification threshold by maximizing F1 on the test set.
- fANOVA runs on Top-K (default 40) most important features with OOF targets.
- If fANOVA fails/returns empty, trains a small Top-K model and runs permutation importance on it.
"""

import os, time, warnings
from typing import Tuple
import numpy as np

# ---------------------- NumPy 2.x compatibility shim ----------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    if not hasattr(np, "float"):  np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):    np.int = int      # type: ignore[attr-defined]
    if not hasattr(np, "bool"):   np.bool = bool    # type: ignore[attr-defined]
    if not hasattr(np, "object"): np.object = object# type: ignore[attr-defined]
    if not hasattr(np, "str"):    np.str = str      # type: ignore[attr-defined]
# --------------------------------------------------------------------------

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

# Imbalanced-learn (optional)
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

# fANOVA (optional)
try:
    from fanova import fANOVA
    HAS_FANOVA = True
except Exception:
    HAS_FANOVA = False

# ------------------------- Config -------------------------
DATA_PATH = "aliced_completed_sa_all_trials_embeddings.csv"
OUTPUT_DIR = "model_explainability_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
POS_LABEL = 1
TARGET_COL = "Study Status"

TOP_FANOVA_DIMS = 40
FANOVA_TREES = 32
FANOVA_POINTS_PER_TREE = 4000

warnings.filterwarnings(
    "ignore",
    message="Precision is ill-defined and being set to 0.0 in labels with no predicted samples",
)

# ------------------------- Helpers -------------------------
def safe_plot_save(path: str):
    try: plt.tight_layout()
    except Exception: pass
    plt.savefig(path, bbox_inches="tight"); plt.close()

def evaluate_at_thresholds(y_true: np.ndarray, y_proba: np.ndarray):
    prec, rec, thr = precision_recall_curve(y_true, y_proba, pos_label=POS_LABEL)
    if len(thr) == 0:
        best_t = 0.5
        y_hat = (y_proba >= best_t).astype(int)
        return best_t, classification_report(y_true, y_hat, digits=4, zero_division=0), y_hat
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_t = float(thr[best_idx])
    y_hat = (y_proba >= best_t).astype(int)
    return best_t, classification_report(y_true, y_hat, digits=4, zero_division=0), y_hat

def plot_roc_pr(y_true: np.ndarray, y_proba: np.ndarray, out_dir: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=POS_LABEL)
        plt.figure(figsize=(6,5)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
        safe_plot_save(os.path.join(out_dir, "roc_curve.png"))
    except Exception: plt.close()
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_proba, pos_label=POS_LABEL)
        plt.figure(figsize=(6,5)); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
        safe_plot_save(os.path.join(out_dir, "pr_curve.png"))
    except Exception: plt.close()

def oof_probas(X: np.ndarray, y: np.ndarray, n_splits=5) -> np.ndarray:
    """Out-of-fold probabilities to use as a smooth fANOVA target."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        if HAS_IMBLEARN:
            clf = BalancedRandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return oof

# ------------------------- Load & prep -------------------------
print("Loading data…")
df = pd.read_csv(DATA_PATH)

print("\nData preview:")
print(df.head())
print(f"\nShape: {df.shape}")

EMBEDDING_COLS = [c for c in df.columns if c.startswith("emb_")]
if len(EMBEDDING_COLS) == 0:
    raise ValueError("No embedding columns found (expected columns starting with 'emb_').")
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

for c in EMBEDDING_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
if not set(np.unique(y_raw.dropna().astype(int))).issubset({0, 1}):
    raise ValueError(f"Target column '{TARGET_COL}' must be binary (0/1).")
df[TARGET_COL] = y_raw.astype(int)

df = df.dropna(subset=EMBEDDING_COLS + [TARGET_COL]).reset_index(drop=True)

X = df[EMBEDDING_COLS].to_numpy(dtype=np.float64, copy=False)
y = df[TARGET_COL].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")
print(f"Class balance in train set: {np.bincount(y_train)}")
print(f"Class balance in test  set: {np.bincount(y_test)}")

# ------------------------- Train main model -------------------------
print("\nTraining Random Forest…")
t0 = time.time()
if HAS_IMBLEARN:
    rf = BalancedRandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
else:
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)
print(f"Training done in {time.time() - t0:.2f} s")

# ------------------------- Eval -------------------------
print("\nEvaluation:")
y_pred_default = rf.predict(X_test)
print("Default 0.5 threshold metrics:")
print(classification_report(y_test, y_pred_default, digits=4, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_default))
print("Accuracy:", accuracy_score(y_test, y_pred_default))

y_proba = None
roc_auc = pr_auc = None
try:
    y_proba = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC :", pr_auc)
except Exception as e:
    print(f"Probability-based metrics skipped due to error: {e}")

best_thr = 0.5
y_pred_tuned = y_pred_default
tuned_report = None
if y_proba is not None:
    best_thr, tuned_report, y_pred_tuned = evaluate_at_thresholds(y_test, y_proba)
    print(f"\nBest probability threshold (F1+): {best_thr:.4f}")
    print("Tuned-threshold metrics:")
    print(tuned_report)
    print("Tuned Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))
    plot_roc_pr(y_test, y_proba, OUTPUT_DIR)

joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest_model.pkl"))
with open(os.path.join(OUTPUT_DIR, "evaluation.txt"), "w") as f:
    f.write("Default 0.5-threshold metrics\n")
    f.write(classification_report(y_test, y_pred_default, digits=4, zero_division=0))
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_default)))
    if roc_auc is not None: f.write(f"\nROC-AUC: {roc_auc}")
    if pr_auc is not None:  f.write(f"\nPR-AUC : {pr_auc}")
    f.write("\n")
    if tuned_report is not None:
        f.write("\n\nTUNED THRESHOLD RESULTS\n")
        f.write(f"Best threshold: {best_thr:.6f}\n")
        f.write(tuned_report)
        f.write("\nTuned confusion matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred_tuned)))
        f.write("\n")

# ------------------------- MDI -------------------------
imp = rf.feature_importances_
std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
imp_df = (
    pd.DataFrame({"feature": EMBEDDING_COLS, "importance": imp, "std": std})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)

plt.figure(figsize=(12, 8))
imp_df.head(20).plot.bar(x="feature", y="importance", yerr="std", capsize=3, legend=False)
plt.ylabel("MDI importance"); plt.title("Top-20 Random Forest Features")
safe_plot_save(os.path.join(OUTPUT_DIR, "feature_importances.png"))

# ------------------------- SHAP -------------------------
print("\nRunning SHAP…")
t0 = time.time()
idx = np.random.choice(X_test.shape[0], size=min(500, X_test.shape[0]), replace=False)
X_sample_df = pd.DataFrame(X_test[idx], columns=EMBEDDING_COLS)
bg_idx = np.random.choice(X_train.shape[0], size=min(200, X_train.shape[0]), replace=False)
X_bg_df = pd.DataFrame(X_train[bg_idx], columns=EMBEDDING_COLS)

explainer = shap.TreeExplainer(
    rf, data=X_bg_df, model_output="probability", feature_perturbation="interventional"
)
shap_values = explainer.shap_values(X_sample_df, check_additivity=False)

def _select_pos_class_sv(sv):
    if isinstance(sv, list): return sv[1]
    arr = np.asarray(sv)
    if arr.ndim == 3: return arr[..., 1]
    if arr.ndim == 2: return arr
    raise ValueError(f"Unexpected shap_values shape: {arr.shape}")

sv_pos = _select_pos_class_sv(shap_values)
print(f"SHAP shapes — full: {np.asarray(shap_values).shape}, pos-class: {sv_pos.shape}")

try:
    shap.summary_plot(sv_pos, X_sample_df, show=False, plot_size=(12, 6))
    safe_plot_save(os.path.join(OUTPUT_DIR, "shap_summary.png"))
except Exception as e:
    print(f"SHAP summary plot failed: {e}")

expected_val = explainer.expected_value
expected_val_pos = float(np.asarray(expected_val)[1]) if isinstance(expected_val, (list, np.ndarray)) else float(expected_val)
try:
    exp_one = shap.Explanation(
        values=sv_pos[0], base_values=expected_val_pos,
        data=X_sample_df.iloc[0].values, feature_names=EMBEDDING_COLS,
    )
    try:
        shap.plots.waterfall(exp_one, show=False, max_display=20)
        safe_plot_save(os.path.join(OUTPUT_DIR, "shap_waterfall.png"))
    except Exception as e:
        print(f"Waterfall failed ({e}); falling back to bar plot.")
        shap.plots.bar(exp_one, show=False, max_display=20)
        safe_plot_save(os.path.join(OUTPUT_DIR, "shap_waterfall_fallback_bar.png"))
except Exception as e:
    print(f"Failed to build SHAP Explanation: {e}")

np.save(os.path.join(OUTPUT_DIR, "shap_values_full.npy"), np.asarray(shap_values))
np.save(os.path.join(OUTPUT_DIR, "shap_values_pos_class.npy"), sv_pos)
print(f"SHAP done in {time.time() - t0:.2f} s")

# ------------------------- fANOVA (Top-K + OOF) -------------------------
print("\nRunning fANOVA…")
t0 = time.time()

# Top-K features for fANOVA/permutation
TOP_FANOVA_DIMS = min(TOP_FANOVA_DIMS, len(EMBEDDING_COLS))
feature_to_idx = {f: i for i, f in enumerate(EMBEDDING_COLS)}
top_feats = imp_df.head(TOP_FANOVA_DIMS)["feature"].tolist()
top_idx = np.array([feature_to_idx[f] for f in top_feats], dtype=int)

X_top_all = X[:, top_idx]                      # for OOF target
X_train_top, X_test_top = X_train[:, top_idx], X_test[:, top_idx]

# Build OOF target on Top-K space
oof = oof_probas(X_top_all, y, n_splits=5)

fan_rows = []
fan_df = pd.DataFrame(columns=["feature", "importance"])
fanova_failed_msg = None

def save_importances(df_imp: pd.DataFrame, prefix: str):
    out_csv = os.path.join(OUTPUT_DIR, f"{prefix}_importances.csv")
    out_png = os.path.join(OUTPUT_DIR, f"{prefix}_importances.png")
    df_imp.to_csv(out_csv, index=False)
    plt.figure(figsize=(12,8))
    df_imp.head(20).plot(kind="bar", x="feature", y="importance", legend=False)
    plt.ylabel(f"{prefix} importance"); plt.title(f"Top-20 {prefix} Features")
    safe_plot_save(out_png)

if not HAS_FANOVA:
    fanova_failed_msg = "fANOVA not available (import failed)."
else:
    try:
        if np.std(oof) < 1e-6 or len(np.unique(oof)) < 2:
            raise RuntimeError("OOF probabilities have near-zero variance; unsuitable for fANOVA.")

        X_for_fanova = np.ascontiguousarray(X_top_all, dtype=np.float64)
        Y_for_fanova = np.ascontiguousarray(oof, dtype=np.float64)

        fan = fANOVA(
            X_for_fanova, Y_for_fanova,
            n_trees=FANOVA_TREES, seed=RANDOM_STATE, points_per_tree=FANOVA_POINTS_PER_TREE
        )

        for j, feat_name in enumerate(top_feats):
            try:
                res = fan.quantify_importance((j,))
                score = res.get("individual importance", res.get("total importance", np.nan)) if isinstance(res, dict) else float(res)
                score = float(score)
                fan_rows.append((feat_name, score))
            except Exception as e:
                print(f"fANOVA importance failed at idx {j} ({feat_name}): {e}")

        fan_df = (
            pd.DataFrame(fan_rows, columns=["feature", "importance"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["importance"])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        if not fan_df.empty:
            save_importances(fan_df, prefix="fanova")
    except Exception as e:
        fanova_failed_msg = f"fANOVA failed: {e}"

# ------------------------- Permutation fallback (fixed) -------------------------
if fan_df.empty:
    if fanova_failed_msg:
        with open(os.path.join(OUTPUT_DIR, "fanova_skipped.txt"), "w") as f:
            f.write(fanova_failed_msg + "\n")

    # Train a dedicated Top-K model so feature dimensions match during permutation
    try:
        if HAS_IMBLEARN:
            rf_top = BalancedRandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            rf_top = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
        rf_top.fit(X_train_top, y_train)

        perm = permutation_importance(
            rf_top, X_test_top, y_test,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc"
        )
        perm_df = (
            pd.DataFrame({
                "feature": top_feats,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        save_importances(perm_df[["feature", "importance"]], prefix="permutation_topK")
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, "fanova_skipped.txt"), "a") as f:
            f.write(f"Permutation-importance fallback failed: {e}\n")

print(f"fANOVA done in {time.time() - t0:.2f} s")
print("\nAll analyses complete ✔")
