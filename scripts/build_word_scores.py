import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "more",
    "most",
    "no",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "under",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
    "would",
    "you",
    "your",
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")


def tokenize(text):
    if not isinstance(text, str):
        return []
    tokens = TOKEN_RE.findall(text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def cosine_similarity(left, right):
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def build_word_scores(ctg_csv, embeddings_csv, completed_ctg_csv, completed_embeddings_csv, output_path, min_count):
    dataset_pairs = [
        (ctg_csv, embeddings_csv, 0),
        (completed_ctg_csv, completed_embeddings_csv, 1),
    ]

    emb_map = {}
    label_map = {}
    emb_cols = None

    for _, (ctg_path, embeddings_path, label) in enumerate(dataset_pairs):
        emb_df = pd.read_csv(embeddings_path)
        if "nct_id" not in emb_df.columns:
            raise ValueError("Embeddings CSV must include nct_id column.")

        if emb_cols is None:
            emb_cols = [col for col in emb_df.columns if col.startswith("emb_")]
            if not emb_cols:
                raise ValueError("No embedding columns found in embeddings CSV.")

        for _, row in emb_df.iterrows():
            nct_id = row.get("nct_id")
            if not nct_id or nct_id in emb_map:
                continue
            emb_map[nct_id] = row[emb_cols].to_numpy(dtype=float)
            label_map[nct_id] = label

    if not emb_map:
        raise ValueError("No embeddings found to build word scores.")

    labeled_embeddings = np.vstack(list(emb_map.values()))
    sentiments = np.array(list(label_map.values()), dtype=int)

    success_mask = sentiments == 1
    if success_mask.sum() == 0 or (~success_mask).sum() == 0:
        raise ValueError("Both COMPLETED and non-COMPLETED trials are required.")

    global_success_rate = float(success_mask.mean())
    success_centroid = labeled_embeddings[success_mask].mean(axis=0)
    failure_centroid = labeled_embeddings[~success_mask].mean(axis=0)

    text_fields = [
        "Study Title",
        "Brief Summary",
        "Conditions",
        "Interventions",
        "Primary Outcome Measures",
        "Secondary Outcome Measures",
        "Sponsor",
        "Collaborators",
        "Phases",
        "Funder Type",
        "Study Type",
        "Study Design",
    ]

    word_stats = {}
    emb_dim = labeled_embeddings.shape[1]

    for ctg_path, _, label in dataset_pairs:
        ctg_df = pd.read_csv(ctg_path)
        ctg_df = ctg_df.rename(columns={"NCT Number": "nct_id"})

        for _, row in ctg_df.iterrows():
            nct_id = row.get("nct_id")
            if nct_id not in emb_map:
                continue
            emb = emb_map[nct_id]
            sentiment = label

            combined_text = " ".join(str(row.get(field, "")) for field in text_fields)
            tokens = set(tokenize(combined_text))
            if not tokens:
                continue

            for token in tokens:
                stats = word_stats.get(token)
                if stats is None:
                    stats = {
                        "count": 0,
                        "success": 0,
                        "sum_emb": np.zeros(emb_dim, dtype=float),
                    }
                    word_stats[token] = stats

                stats["count"] += 1
                stats["success"] += int(sentiment == 1)
                stats["sum_emb"] += emb

    output = {
        "meta": {
            "min_count": min_count,
            "global_success_rate": round(global_success_rate, 6),
            "total_trials": int(len(emb_map)),
            "label_source": "ctg-studies + ctg-studies-completed",
            "success_definition": "COMPLETED dataset",
        },
        "data": {},
    }

    for word, stats in word_stats.items():
        count = stats["count"]
        if count < min_count:
            continue
        success_rate = stats["success"] / count
        mean_emb = stats["sum_emb"] / count
        assoc_score = cosine_similarity(mean_emb, success_centroid) - cosine_similarity(
            mean_emb, failure_centroid
        )

        output["data"][word] = {
            "count": int(count),
            "success_rate": round(success_rate, 6),
            "lift": round(success_rate - global_success_rate, 6),
            "assoc_score": round(assoc_score, 6),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build word association scores for the static site."
    )
    parser.add_argument(
        "--ctg-csv",
        default="ctg-studies.csv",
        help="Path to ctg-studies.csv.",
    )
    parser.add_argument(
        "--embeddings-csv",
        default="labeled_sa_all_trials_embeddings.csv",
        help="Path to labeled embeddings CSV for ctg-studies.csv.",
    )
    parser.add_argument(
        "--completed-ctg-csv",
        default="ctg-studies-completed.csv",
        help="Path to ctg-studies-completed.csv.",
    )
    parser.add_argument(
        "--completed-embeddings-csv",
        default="labeled_completed_sa_all_trials_embeddings.csv",
        help="Path to labeled embeddings CSV for ctg-studies-completed.csv.",
    )
    parser.add_argument(
        "--output",
        default="site/word_scores.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum number of trials containing the word.",
    )

    args = parser.parse_args()

    build_word_scores(
        Path(args.ctg_csv),
        Path(args.embeddings_csv),
        Path(args.completed_ctg_csv),
        Path(args.completed_embeddings_csv),
        Path(args.output),
        args.min_count,
    )


if __name__ == "__main__":
    main()
