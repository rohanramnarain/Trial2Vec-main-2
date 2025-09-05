import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Load data
df = pd.read_csv("all_trials_embeddings.csv")

# 2. Separate trial IDs (if the first column is an ID) and keep only numeric embedding columns
trial_ids = df.iloc[:, 0]             # This is the 'trial_id' column if it's first
embeddings = df.iloc[:, 1:].values    # numeric columns only

# 3. Normalize (for cosine-like approach)
X_norm = normalize(embeddings, norm='l2')

# 4. Silhouette for multiple k
Ks = range(2, 9)
silhouette_scores = []

for k in Ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_norm)
    score = silhouette_score(X_norm, labels)
    silhouette_scores.append(score)
    print(f"k={k}, Silhouette={score:.4f}")

# 5. Plot silhouette scores
plt.figure()
plt.plot(Ks, silhouette_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score") #do this first to figure out best num of clusters 
plt.title("Silhouette Scores")
plt.show() #returns the best k best number of clusters 

# 6. Best k
best_k = Ks[np.argmax(silhouette_scores)]
print(f"Best k: {best_k}")

# 7. Final clustering with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42)
final_labels = kmeans.fit_predict(X_norm)

# Attach cluster labels back to the original df
df["cluster"] = final_labels

# 8. Similarity matrix (cosine)
similarity_matrix = cosine_similarity(X_norm) #then make the clusters here 

# 9. Save the results
df.to_csv("output_all_trials_embeddings_with_clusters.csv", index=False)

# -----------------------------------------------------
#  Additional: PCA VISUALIZATION IN 2D AND 3D
# -----------------------------------------------------

# ---------- 2D PCA Visualization ---------- 
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_norm)  # shape: (31, 2)

plt.figure()
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=final_labels)
plt.title("2D PCA of Trials")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()  # shows cluster assignments as a color scale 
plt.show()

# Optional text labels for each point:
plt.figure()
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=final_labels)
for i, txt in enumerate(trial_ids):
    plt.text(X_pca_2d[i, 0], X_pca_2d[i, 1], str(txt))
plt.title("2D PCA of Trials (labeled)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()


# ---------- 3D PCA Visualization ----------
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_norm)  # shape: (31, 3)

# We'll create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the three principal components
sc = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                c=final_labels)

# Label axes
ax.set_title("3D PCA of Trials")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")

# Add color bar to reflect the cluster labels
plt.colorbar(sc)
plt.show()

# If you also want to label each point by trial ID in 3D:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                c=final_labels)
for i, txt in enumerate(trial_ids):
    ax.text(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], str(txt))
ax.set_title("3D PCA of Trials (labeled)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.colorbar(sc)
plt.show()
