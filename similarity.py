import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Specify the path to your CSV file
csv_path = "all_trials_embeddings.csv"  # change to your actual filename/path

# 2. Read the CSV into a pandas DataFrame
df = pd.read_csv(csv_path)

# 3. Identify columns that contain your embedding components
#    i.e., emb_0 through emb_127
embedding_cols = [col for col in df.columns if col.startswith("emb_")]

# 4. Extract just those embedding columns into a NumPy array
embedding_matrix = df[embedding_cols].values

# 5. Compute pairwise cosine similarity (matrix size N x N, where N is number of rows)
similarity_matrix = cosine_similarity(embedding_matrix)

# 6. Build a new DataFrame with every pair (i, j) and their similarity
rows = []
num_rows = len(df)
for i in range(num_rows):
    for j in range(num_rows):
        rows.append({
            "nct_id_i": df.loc[i, "nct_id"],
            "nct_id_j": df.loc[j, "nct_id"],
            "similarity": similarity_matrix[i, j]
        })

df_similarities = pd.DataFrame(rows)

# 7. Write to a CSV file
output_path = "pairwise_similarities.csv"
df_similarities.to_csv(output_path, index=False)

print(f"Saved pairwise similarities to {output_path}")
print("Preview of the output:")
print(df_similarities.head(10))
