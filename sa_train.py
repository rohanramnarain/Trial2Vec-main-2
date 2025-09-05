#!/usr/bin/env python3
"""
Train a binary sentiment‑style classifier for clinical‑trial embeddings.

The CSV is expected to contain one row per trial with columns:
  * emb_0 … emb_127   –  the 128‑D Trial2Vec embedding
  * <label column>    –  1 = negative/terminated, 0 = positive/ongoing (default name: "sentiment")
  * any other columns are ignored

Example
-------
$ python train_sentiment.py \
    --csv combined_sa_all_trials_embeddings.csv \
    --label_col sentiment \
    --model_out sentiment_model.joblib
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import dump


def parse_args():
    p = argparse.ArgumentParser(description="Train sentiment classifier on trial embeddings")
    p.add_argument("--csv", required=True, help="Path to CSV with embeddings + label column")
    p.add_argument("--label_col", default="Sentiment", help="Name of the binary label column (1 = bad, 0 = good)")
    p.add_argument("--test_size", type=float, default=0.2, help="Hold‑out fraction for test set [0‑1]")
    p.add_argument("--model_out", default="sentiment_model.joblib", help="Where to save the fitted model")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # 1) Load data ---------------------------------------------------------
    df = pd.read_csv(csv_path)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in {csv_path.name}")

    embed_cols = [c for c in df.columns if c.startswith("emb_")]
    if not embed_cols:
        raise ValueError("No embedding columns (emb_*) found in CSV.")

    X = df[embed_cols].values
    y = df[args.label_col].values

    # 2) split -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # 3) pipeline ----------------------------------------------------------
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipe.fit(X_train, y_train)

    # 4) evaluation --------------------------------------------------------
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
    print("ROC‑AUC: {:.3f}".format(roc_auc_score(y_test, y_proba)))

    # 5) save model --------------------------------------------------------
    dump(pipe, args.model_out)
    print(f"\nModel saved to {args.model_out}\nDone.")


if __name__ == "__main__":
    main()
