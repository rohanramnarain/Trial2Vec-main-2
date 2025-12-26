"""
Utility to (1) map SHAP top dimensions back to the trials that drive them and
(2) train lightweight linear probes that attach human-readable tags to the
latent Trial2Vec dimensions.

Outputs (under model_explainability_outputs/):
  - shap_dim_extremes.csv    : top SHAP dims with highest/lowest trials
  - shap_dim_summary.md      : quick textual rundown
  - probe_*.csv / probe_*.md : per-tag linear probe coefficients and metrics
    - probe_coefficients.png   : compact bar charts of strongest probe weights

This replays the exact train/test split and SHAP sampling used in
clintrial_fixed.py (same RNG seed), so the loaded shap_values_pos_class.npy
aligns with the rows inspected here.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("aliced_completed_sa_all_trials_embeddings.csv")
META_PATH = Path("ctg-studies-completed.csv")
OUTPUT_DIR = Path("model_explainability_outputs")
SHAP_PATH = OUTPUT_DIR / "shap_values_pos_class.npy"
CTGOV_CACHE = OUTPUT_DIR / "nct_cache.json"

SHAP_SAMPLE = 200
SHAP_BG = 50
POS_LABEL = 1


@dataclass
class ShapDimSummary:
    dim: int
    mean_abs: float
    trials: pd.DataFrame


def _load_embeddings() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(DATA_PATH)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found")
    df["Study Status"] = pd.to_numeric(df["Study Status"], errors="coerce").astype(int)
    df = df.dropna(subset=emb_cols + ["Study Status"]).reset_index(drop=True)
    return df, emb_cols


def _load_local_trials() -> pd.DataFrame:
    rows = []
    for p in Path(".").glob("NCT*.csv"):
        try:
            tdf = pd.read_csv(p)
        except Exception:
            continue
        if "NCT Number" not in tdf.columns:
            continue
        # Take first row (single-trial file)
        r = tdf.iloc[0]
        rows.append(
            {
                "nct_id": str(r.get("NCT Number", "")),
                "Study Title": r.get("Study Title", np.nan),
                "Brief Summary": r.get("Brief Summary", np.nan),
                "Conditions": r.get("Conditions", np.nan),
                "Phases": r.get("Phases", np.nan),
                "Study Type": r.get("Study Type", np.nan),
                "Funder Type": r.get("Funder Type", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _load_cache() -> Dict[str, Dict[str, str]]:
    if CTGOV_CACHE.exists():
        try:
            return json.loads(CTGOV_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Dict[str, str]]):
    CTGOV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CTGOV_CACHE.write_text(json.dumps(cache, indent=2))


def _fetch_ctgov(nct_id: str, timeout: float = 10.0) -> Dict[str, str]:
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    params = {
        "fields": "protocolSection.identificationModule.briefTitle,protocolSection.descriptionModule.briefSummary,protocolSection.conditionsModule,protocolSection.designModule",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        ps = js.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        desc = ps.get("descriptionModule", {})
        cond = ps.get("conditionsModule", {})
        design = ps.get("designModule", {})
        cond_list = cond.get("conditions", [])
        phase_val = design.get("phase") or design.get("phases")
        if isinstance(phase_val, list) and phase_val:
            phase_val = phase_val[0]
        return {
            "Study Title": ident.get("briefTitle"),
            "Brief Summary": desc.get("briefSummary"),
            "Conditions": "|".join(cond_list) if cond_list else None,
            "Phases": phase_val,
            "Study Type": design.get("studyType"),
            "Funder Type": None,
        }
    except Exception:
        return {}


def _replay_split(df: pd.DataFrame, emb_cols: List[str]):
    X = df[emb_cols].to_numpy(dtype=np.float64, copy=False)
    y = df["Study Status"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Reproduce SHAP sampling (same code as clintrial_fixed.py)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(X_test.shape[0], size=min(SHAP_SAMPLE, X_test.shape[0]), replace=False)
    bg_idx = rng.choice(X_train.shape[0], size=min(SHAP_BG, X_train.shape[0]), replace=False)

    # Keep mapping to original dataframe rows
    # train/test come from deterministic split; build indices
    df_idx = np.arange(len(df))
    _, test_idx = train_test_split(
        df_idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    X_sample_df = pd.DataFrame(X_test[idx], columns=emb_cols)
    X_bg_df = pd.DataFrame(X_train[bg_idx], columns=emb_cols)
    sample_df_indices = test_idx[idx]

    return X_sample_df, X_bg_df, sample_df_indices


def _load_shap() -> np.ndarray:
    arr = np.load(SHAP_PATH, allow_pickle=True)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D shap array, got {arr.shape}")
    return arr


def _make_snippet(src: pd.Series) -> str:
    title = str(src.get("Study Title", ""))
    cond = str(src.get("Conditions", ""))
    brief = str(src.get("Brief Summary", ""))
    pieces = [p.strip() for p in [title, cond, brief] if p and p.strip()]
    snippet = " | ".join(pieces)
    if len(snippet) > 240:
        snippet = snippet[:240] + "…"
    return snippet


def build_top_shap_dim_extremes(
    shap_vals: np.ndarray,
    X_sample_df: pd.DataFrame,
    sample_df_indices: np.ndarray,
    full_df: pd.DataFrame,
    top_k_dims: int = 20,
    per_side: int = 8,
) -> List[ShapDimSummary]:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_dims = np.argsort(mean_abs)[::-1][:top_k_dims]
    results: List[ShapDimSummary] = []

    for d in top_dims:
        sv = shap_vals[:, d]
        feature_vals = X_sample_df.iloc[:, d]
        order = np.argsort(sv)
        low_idx = order[:per_side]
        high_idx = order[-per_side:][::-1]

        def rows(idxs, direction: str) -> pd.DataFrame:
            rows_list = []
            for i in idxs:
                global_row = int(sample_df_indices[i])
                src = full_df.iloc[global_row]
                rows_list.append(
                    {
                        "nct_id": src.get("nct_id", np.nan),
                        "study_status": int(src["Study Status"]),
                        "shap_value": float(sv[i]),
                        "feature_value": float(feature_vals.iloc[i]),
                        "direction": direction,
                        "snippet": _make_snippet(src),
                    }
                )
            return pd.DataFrame(rows_list)

        combined = pd.concat([rows(high_idx, "high"), rows(low_idx, "low")], ignore_index=True)
        results.append(
            ShapDimSummary(dim=int(d), mean_abs=float(mean_abs[d]), trials=combined)
        )

    return results


def save_shap_reports(summaries: List[ShapDimSummary]):
    rows = []
    for s in summaries:
        for _, r in s.trials.iterrows():
            r_dict = r.to_dict()
            r_dict.update({"dim": s.dim, "mean_abs": s.mean_abs})
            rows.append(r_dict)
    out_csv = OUTPUT_DIR / "shap_dim_extremes.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Compact Markdown summary
    lines = ["# SHAP dimension extremes", "", "Top dimensions by mean |SHAP|:"]
    for s in summaries:
        lines.append(f"- dim {s.dim}: mean|shap|={s.mean_abs:.4f}")
    lines.append("")
    lines.append("Each dimension lists trials with highest/lowest SHAP contributions.")
    md_path = OUTPUT_DIR / "shap_dim_summary.md"
    md_path.write_text("\n".join(lines))


def _clean_phase(val: str) -> str:
    if not isinstance(val, str):
        return "NA"
    v = val.upper().replace("PHASE ", "PHASE").replace("EARLY", "0")
    tokens = re.split(r"[/|]", v)
    nums = []
    for t in tokens:
        m = re.search(r"PHASE\s*([0-4])", t)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return "NA"
    best = max(nums)
    return f"PHASE{best}"


def _condition_head(val: str) -> str:
    if not isinstance(val, str):
        return "NA"
    parts = [p.strip() for p in val.split("|") if p.strip()]
    if not parts:
        return "NA"
    return parts[0].upper()


def _prepare_meta(df_emb: pd.DataFrame, target_ids: set | None = None) -> pd.DataFrame:
    if not META_PATH.exists():
        raise FileNotFoundError("ctg-studies-completed.csv not found for metadata join")

    meta = pd.read_csv(META_PATH)
    meta = meta.rename(columns={"NCT Number": "nct_id"})
    meta["nct_id"] = meta["nct_id"].astype(str)
    df_emb["nct_id"] = df_emb["nct_id"].astype(str)
    local_trials = _load_local_trials()

    merged = df_emb.merge(meta, on="nct_id", how="left", suffixes=("", "_meta"))
    if not local_trials.empty:
        merged = merged.merge(local_trials, on="nct_id", how="left", suffixes=("", "_local"))
        for col in ["Study Title", "Brief Summary", "Conditions", "Phases", "Study Type", "Funder Type"]:
            if col in merged.columns and f"{col}_local" in merged.columns:
                merged[col] = merged[col].fillna(merged[f"{col}_local"])

    # Fill gaps via ClinicalTrials.gov API (cached)
    cache = _load_cache()
    missing_mask = merged["Study Title"].isna() | merged["Brief Summary"].isna()
    missing_ids = merged.loc[missing_mask, "nct_id"].dropna().unique().tolist()
    if target_ids is not None:
        missing_ids = [i for i in missing_ids if i in target_ids]
    for nct_id in missing_ids:
        if nct_id in cache:
            payload = cache[nct_id]
        else:
            payload = _fetch_ctgov(nct_id)
            if payload:
                cache[nct_id] = payload
                time.sleep(0.2)  # light pacing
        if not payload:
            continue
        for k, v in payload.items():
            if v is None:
                continue
            if k in merged.columns:
                merged.loc[merged["nct_id"] == nct_id, k] = merged.loc[merged["nct_id"] == nct_id, k].fillna(v)
    if cache:
        _save_cache(cache)

    merged["phase_clean"] = merged["Phases"].apply(_clean_phase) if "Phases" in merged.columns else "NA"
    merged["condition_head"] = merged["Conditions"].apply(_condition_head) if "Conditions" in merged.columns else "NA"
    return merged


def _train_probe(
    X: np.ndarray,
    y: np.ndarray,
    min_count: int = 3,
    tag: str = "tag",
    multi_ok: bool = True,
) -> Dict:
    # filter rare classes
    labels, counts = np.unique(y, return_counts=True)
    keep = labels[counts >= min_count]
    mask = np.isin(y, keep)
    X_f = X[mask]
    y_f = y[mask]
    if len(np.unique(y_f)) < 2:
        return {"tag": tag, "skipped": "not_enough_classes"}

    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="auto" if multi_ok else "ovr",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    clf.fit(X_f, y_f)
    y_pred = clf.predict(X_f)
    metrics = {
        "acc": float(accuracy_score(y_f, y_pred)),
        "f1_macro": float(f1_score(y_f, y_pred, average="macro")),
        "n_samples": int(len(y_f)),
        "n_classes": int(len(np.unique(y_f))),
    }

    coefs = clf.named_steps["clf"].coef_
    if coefs.ndim == 1:
        coefs = coefs[None, :]
    classes = clf.named_steps["clf"].classes_

    coef_rows = []
    for cls, row in zip(classes, coefs):
        top_pos = np.argsort(row)[::-1][:10]
        top_neg = np.argsort(row)[:10]
        coef_rows.append(
            {
                "class": cls,
                "top_pos_dims": ",".join(map(str, top_pos.tolist())),
                "top_neg_dims": ",".join(map(str, top_neg.tolist())),
            }
        )

    res = {"tag": tag, "metrics": metrics, "coef_rows": coef_rows}
    res["coef_matrix"] = coefs
    res["classes"] = classes
    return res


def plot_probe_coefficients(reports: List[Dict], emb_cols: List[str], top_k: int = 8):
    """Small visualization of strongest probe weights per tag."""

    plot_ready = [r for r in reports if "skipped" not in r and "coef_matrix" in r]
    if not plot_ready:
        return

    n_rows = len(plot_ready)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, r in zip(axes, plot_ready):
        coef = np.asarray(r.get("coef_matrix"))
        if coef.ndim == 1:
            coef = coef[None, :]
        mean_mag = np.mean(np.abs(coef), axis=0)
        top_idx = np.argsort(mean_mag)[::-1][: min(top_k, len(mean_mag))]
        vals = mean_mag[top_idx][::-1]
        labels = [emb_cols[i] if i < len(emb_cols) else f"dim_{i}" for i in top_idx][::-1]

        ax.barh(np.arange(len(top_idx)), vals, color="#3a6ea5")
        ax.set_yticks(np.arange(len(top_idx)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("mean |coef| (standardized space)")

        metrics = r.get("metrics", {})
        acc = metrics.get("acc")
        f1 = metrics.get("f1_macro")
        title_bits = [r.get("tag", "probe")]
        if acc is not None and f1 is not None:
            title_bits.append(f"f1={f1:.2f}, acc={acc:.2f}")
        ax.set_title(" | ".join(title_bits))
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "probe_coefficients.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_probes(merged: pd.DataFrame, emb_cols: List[str]):
    X = merged[emb_cols].to_numpy()

    probes = []

    if "Study Type" in merged.columns:
        probes.append(("study_type", merged["Study Type"].fillna("NA").astype(str).str.upper().to_numpy(), True))

    if "phase_clean" in merged.columns:
        probes.append(("phase", merged["phase_clean"].to_numpy(), True))

    if "Funder Type" in merged.columns:
        funder = merged["Funder Type"].fillna("NA").astype(str).str.upper().to_numpy()
        probes.append(("funder", funder, True))

    if "condition_head" in merged.columns:
        probes.append(("condition", merged["condition_head"].to_numpy(), True))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reports = []
    for tag, y_raw, multi_ok in probes:
        res = _train_probe(X, y_raw, min_count=3, tag=tag, multi_ok=multi_ok)
        if "skipped" in res:
            reports.append(res)
            continue
        probe_csv = OUTPUT_DIR / f"probe_{tag}.csv"
        pd.DataFrame(res["coef_rows"]).to_csv(probe_csv, index=False)
        reports.append(res)

    plot_probe_coefficients(reports, emb_cols, top_k=8)

    md_lines = ["# Linear probe summary", ""]
    for r in reports:
        if "skipped" in r:
            md_lines.append(f"- {r['tag']}: skipped ({r['skipped']})")
        else:
            m = r["metrics"]
            md_lines.append(
                f"- {r['tag']}: acc={m['acc']:.3f}, f1_macro={m['f1_macro']:.3f}, n={m['n_samples']}, classes={m['n_classes']}"
            )
    (OUTPUT_DIR / "probe_summary.md").write_text("\n".join(md_lines))

    return reports


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, emb_cols = _load_embeddings()
    X_sample_df, X_bg_df, sample_df_indices = _replay_split(df, emb_cols)
    target_ids = set(df.loc[sample_df_indices, "nct_id"].astype(str).tolist())
    merged = _prepare_meta(df.copy(), target_ids=target_ids)
    shap_vals = _load_shap()

    summaries = build_top_shap_dim_extremes(
        shap_vals=shap_vals,
        X_sample_df=X_sample_df,
        sample_df_indices=sample_df_indices,
        full_df=merged,
        top_k_dims=20,
        per_side=8,
    )
    save_shap_reports(summaries)

    run_probes(merged, emb_cols)

    print("Reports written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
