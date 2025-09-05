#!/usr/bin/env python3
"""
Train a Random Forest on trial-embedding data and run several
explainability analyses (MDI feature importance, SHAP, fANOVA).

Patches in this revision
------------------------
1. **Correct label column** – `y` is taken from the “Study Status” field.
2. **SHAP shape-mismatch fix** – pass a *DataFrame* (not a NumPy array)
   to `TreeExplainer` and the plotting functions so the number of columns
   matches the SHAP matrix.
3. Minor: prints for sanity-checking shapes and class balance.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from fanova import fANOVA

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_PATH = "aliced_completed_sa_all_trials_embeddings.csv"  # <-- adjust
OUTPUT_DIR = "model_explainability_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
print("Loading data…")
df = pd.read_csv(DATA_PATH)

print("\nData preview:")
print(df.head())
print(f"\nShape: {df.shape}")

# ---------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------
TARGET_COL = "Study Status"
EMBEDDING_COLS = [c for c in df.columns if c.startswith("emb_")]

X = df[EMBEDDING_COLS].values
y = df[TARGET_COL].astype(int).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")
print(f"Class balance in train set: {np.bincount(y_train)}")
print(f"Class balance in test  set: {np.bincount(y_test)}")

# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------
print("\nTraining Random Forest…")
t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
rf.fit(X_train, y_train)

print(f"Training done in {time.time() - t0:.2f} s")

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
print("\nEvaluation:")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest_model.pkl"))

# ---------------------------------------------------------------------
# MDI Feature importance
# ---------------------------------------------------------------------
imp = rf.feature_importances_
std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
imp_df = (
    pd.DataFrame({"feature": EMBEDDING_COLS, "importance": imp, "std": std})
    .sort_values("importance", ascending=False)
)
imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)

plt.figure(figsize=(12, 8))
imp_df.head(20).plot.bar(
    x="feature", y="importance", yerr="std", capsize=3, legend=False
)
plt.ylabel("MDI importance")
plt.title("Top-20 Random Forest Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"))
plt.close()

# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------
print("\nRunning SHAP…")
t0 = time.time()

# sample ≤500 rows for speed
idx = np.random.choice(X_test.shape[0], size=min(500, X_test.shape[0]), replace=False)
X_sample_df = pd.DataFrame(X_test[idx], columns=EMBEDDING_COLS)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample_df)  # returns list [class0, class1]

# -- summary plot
shap.summary_plot(
    shap_values[1],
    X_sample_df,
    show=False,
    plot_size=(12, 6),
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"))
plt.close()

# -- force plot (first sample, class 1)
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_sample_df.iloc[0],
    matplotlib=True,
    show=False,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_force_plot.png"))
plt.close()

np.save(os.path.join(OUTPUT_DIR, "shap_values.npy"), shap_values)
print(f"SHAP done in {time.time() - t0:.2f} s")

# ---------------------------------------------------------------------
# fANOVA (top-50 for speed)
# ---------------------------------------------------------------------
print("\nRunning fANOVA…")
t0 = time.time()

X_train_df = pd.DataFrame(X_train, columns=EMBEDDING_COLS)
fanova = fANOVA(X_train_df, pd.Series(y_train))

fan_imp = {
    feat: fanova.quantify_importance((feat,))[(feat,)]
    for feat in EMBEDDING_COLS[:50]
}
fan_df = (
    pd.DataFrame.from_dict(fan_imp, orient="index", columns=["importance"])
    .sort_values("importance", ascending=False)
)
fan_df.to_csv(os.path.join(OUTPUT_DIR, "fanova_importances.csv"))

plt.figure(figsize=(12, 8))
fan_df.head(20).plot.bar(y="importance", legend=False)
plt.ylabel("fANOVA importance")
plt.title("Top-20 fANOVA Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fanova_importances.png"))
plt.close()

print(f"fANOVA done in {time.time() - t0:.2f} s")
print("\nAll analyses complete ✔")
