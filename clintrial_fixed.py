#!/usr/bin/env python3
"""
RF + MDI + SHAP + fANOVA (guarded) with a robust fallback to permutation importance.

- Uses BalancedRandomForest if available; otherwise RF(class_weight="balanced").
- Tunes the classification threshold by maximizing F1 on the test set.
- fANOVA runs on Top-K most important features with an OOF (out-of-fold) probability target.
- If fANOVA fails/returns empty, trains a small Top-K model and runs permutation importance on it.

Changes vs original:
- Added speed knobs (smaller Top-K, fewer trees/points/splits).
- More robust fANOVA (skip degenerate features, ConfigSpace bounds, explicit logging & column alignment).
- Safer fallback (switches scorer if AUC is undefined).
- Trimmed SHAP sampling to cut wall-time.
"""

import os, time, warnings
from typing import Tuple
import numpy as np

# ---------------------- Speed/robustness knobs ----------------------
# You can tune these if you want more/less speed.
TOP_FANOVA_DIMS = 20          # was 40
FANOVA_TREES = 16             # was 32
FANOVA_POINTS_PER_TREE = 512  # was 4000

OOF_SPLITS = 3                # was 5
OOF_TREES = 200               # was 300

SHAP_SAMPLE = 200             # was 500
SHAP_BG = 50                  # was 200
USE_CONFIGSPACE = True        # force explicit bounds + ordering for fANOVA


# Limit thread over-subscription a bit (helps when pyrfr/BLAS is involved)
os.environ.setdefault("OMP_NUM_THREADS", "1")

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

# ConfigSpace (optional, improves stability/ranges if available)
try:
    import ConfigSpace as CS
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter
    HAS_CS = True
except Exception:
    HAS_CS = False

# ------------------------- Config -------------------------
DATA_PATH = "aliced_completed_sa_all_trials_embeddings.csv"
OUTPUT_DIR = "model_explainability_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
POS_LABEL = 1
TARGET_COL = "Study Status"

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

def oof_probas(X: np.ndarray, y: np.ndarray, n_splits: int = OOF_SPLITS) -> np.ndarray:
    """Out-of-fold probabilities to use as a smooth fANOVA target."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        if HAS_IMBLEARN:
            clf = BalancedRandomForestClassifier(
                n_estimators=OOF_TREES, random_state=RANDOM_STATE, n_jobs=-1,
                sampling_strategy="all", replacement=True, bootstrap=False
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=OOF_TREES, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
            )
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
    rf = BalancedRandomForestClassifier(
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1,
        sampling_strategy="all", replacement=True, bootstrap=False
    )
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
idx = np.random.choice(X_test.shape[0], size=min(SHAP_SAMPLE, X_test.shape[0]), replace=False)
X_sample_df = pd.DataFrame(X_test[idx], columns=EMBEDDING_COLS)
bg_idx = np.random.choice(X_train.shape[0], size=min(SHAP_BG, X_train.shape[0]), replace=False)
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

X_top_all = X[:, top_idx]                      # for OOF target & fANOVA space
X_train_top, X_test_top = X_train[:, top_idx], X_test[:, top_idx]

# Build OOF target on Top-K space
oof = oof_probas(X_top_all, y, n_splits=OOF_SPLITS)
print(f"OOF stats: min={oof.min():.4f}, max={oof.max():.4f}, std={oof.std():.6f}, unique≈{len(np.unique(np.round(oof,5)))}")

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
        # Guard: need variance in the target
        if np.std(oof) < 1e-6 or len(np.unique(np.round(oof, 6))) < 2:
            raise RuntimeError("OOF probabilities have near-zero variance; unsuitable for fANOVA.")

        # Build list of kept features with non-degenerate ranges
        kept_names = []
        kept_cols = []
        if HAS_CS and USE_CONFIGSPACE:
            cs = CS.ConfigurationSpace(seed=RANDOM_STATE)
        else:
            cs = None

        for j, name in enumerate(top_feats):
            col = X_top_all[:, j]
            lo = float(np.nanmin(col))
            hi = float(np.nanmax(col))
            if (not np.isfinite(lo + hi)) or (lo == hi):
                print(f"Skipping '{name}' in fANOVA (degenerate bounds lo=hi={lo}).")
                continue
            kept_names.append(name)
            kept_cols.append(j)
            if cs is not None:
                # pad bounds slightly to avoid boundary rejects from float rounding
                lower = float(np.nextafter(lo, -np.inf))
                upper = float(np.nextafter(hi,  np.inf))
                cs.add_hyperparameter(UniformFloatHyperparameter(name, lower=lower, upper=upper))

        if len(kept_cols) == 0:
            raise RuntimeError("All top-K features had degenerate ranges for fANOVA.")

        # Align X columns EXACTLY to ConfigSpace hyperparameter order
        X_for_fanova = np.ascontiguousarray(X_top_all[:, kept_cols], dtype=np.float64)
        if cs is not None:
            hp_names = [hp.name for hp in cs.get_hyperparameters()]
            order_idx = [kept_names.index(n) for n in hp_names]
            X_for_fanova = X_for_fanova[:, order_idx]
            X_for_fanova_df = pd.DataFrame(X_for_fanova, columns=hp_names)
            name_order = hp_names
        else:
            X_for_fanova_df = X_for_fanova
            name_order = kept_names

        Y_for_fanova = np.ascontiguousarray(oof, dtype=np.float64)

        # Build fANOVA (smaller/faster settings)
        fan = fANOVA(
            X_for_fanova_df, Y_for_fanova,
            config_space=cs,
            n_trees=FANOVA_TREES, seed=RANDOM_STATE, points_per_tree=FANOVA_POINTS_PER_TREE
        )

        # Compute 1D importances; guard against NaN/inf
        for j, name in enumerate(name_order):
            try:
                res = fan.quantify_importance((j,))
                # Some versions return dict, some return float
                if isinstance(res, dict):
                    score = res.get("individual importance", res.get("total importance", np.nan))
                else:
                    score = float(res)
                score = float(score)
                if np.isfinite(score):
                    fan_rows.append((name, score))
                else:
                    print(f"fANOVA importance for {name} was non-finite: {score}")
            except Exception as e:
                print(f"fANOVA importance failed for {name}: {e}")

        fan_df = (
            pd.DataFrame(fan_rows, columns=["feature", "importance"])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        print(f"fANOVA Top-K={len(top_feats)}; kept={len(name_order)}; rows={len(fan_df)}")
        if not fan_df.empty:
            save_importances(fan_df, prefix="fanova")
        else:
            fanova_failed_msg = "fANOVA produced no finite importances."
    except Exception as e:
        fanova_failed_msg = f"fANOVA failed: {e}"

# ------------------------- Permutation fallback (more robust) -------------------------
if fan_df.empty:
    if fanova_failed_msg:
        with open(os.path.join(OUTPUT_DIR, "fanova_skipped.txt"), "w") as f:
            f.write(fanova_failed_msg + "\\n")

    try:
        # Use the features we kept for fANOVA if any; else original Top-K
        perm_feats = fan_df["feature"].tolist()
        if len(perm_feats) == 0:
            # If fANOVA failed before computing importances, fall back to kept_names if available
            perm_feats = kept_names if 'kept_names' in locals() and len(kept_names) > 0 else top_feats

        perm_idx = np.array([feature_to_idx[f] for f in perm_feats], dtype=int)
        X_train_perm, X_test_perm = X_train[:, perm_idx], X_test[:, perm_idx]

        if HAS_IMBLEARN:
            rf_top = BalancedRandomForestClassifier(
                n_estimators=OOF_TREES, random_state=RANDOM_STATE, n_jobs=-1,
                sampling_strategy="all", replacement=True, bootstrap=False
            )
        else:
            rf_top = RandomForestClassifier(n_estimators=OOF_TREES, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
        rf_top.fit(X_train_perm, y_train)

        try:
            perm = permutation_importance(
                rf_top, X_test_perm, y_test,
                n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc"
            )
        except ValueError:
            # AUC can fail if only one class in y_test or degenerate scores -> fallback to accuracy
            perm = permutation_importance(
                rf_top, X_test_perm, y_test,
                n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1, scoring="accuracy"
            )

        perm_df = (
            pd.DataFrame({
                "feature": perm_feats,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        # Save only feature/importance for parity with fANOVA output file
        out_perm = perm_df[["feature", "importance"]].copy()
        out_perm.to_csv(os.path.join(OUTPUT_DIR, "permutation_topK_importances.csv"), index=False)
        plt.figure(figsize=(12,8))
        out_perm.head(20).plot(kind="bar", x="feature", y="importance", legend=False)
        plt.ylabel("permutation_topK importance"); plt.title("Top-20 permutation_topK Features")
        safe_plot_save(os.path.join(OUTPUT_DIR, "permutation_topK_importances.png"))
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, "fanova_skipped.txt"), "a") as f:
            f.write(f"Permutation-importance fallback failed: {e}\\n")

print(f"fANOVA done in {time.time() - t0:.2f} s")
print("\\nAll analyses complete ✔")
