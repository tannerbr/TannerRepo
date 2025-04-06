import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ----------------------------
# Load CSV Data for Recommendations
# ----------------------------

# Define the path to your collaborative filtering CSV
collab_csv_path = './data/Collaborative_Filtering.csv'
if not os.path.exists(collab_csv_path):
    raise FileNotFoundError(f"Collaborative filtering CSV not found at path: {collab_csv_path}")

# Read the collaborative filtering CSV file
df_collab = pd.read_csv(collab_csv_path)
df_collab.columns = df_collab.columns.str.strip()

# If "ItemID" is not in the CSV, create it from the DataFrame index
if "ItemID" not in df_collab.columns:
    print("Column 'ItemID' not found in collaborative filtering CSV. Creating 'ItemID' from index.")
    df_collab.reset_index(inplace=True)
    df_collab.rename(columns={'index': 'ItemID'}, inplace=True)
df_collab['ItemID'] = df_collab['ItemID'].astype(int)

# Load the content filtering CSV similarly
content_csv_path = './data/content_filtering.csv'
if not os.path.exists(content_csv_path):
    raise FileNotFoundError(f"Content filtering CSV not found at path: {content_csv_path}")

df_content = pd.read_csv(content_csv_path)
df_content.columns = df_content.columns.str.strip()

if "ItemID" not in df_content.columns:
    print("Column 'ItemID' not found in content filtering CSV. Creating 'ItemID' from index.")
    df_content.reset_index(inplace=True)
    df_content.rename(columns={'index': 'ItemID'}, inplace=True)
df_content['ItemID'] = df_content['ItemID'].astype(int)

# ----------------------------
# Build the Available Items List from CSV Data
# ----------------------------
# We assume that df_collab has an "If you liked" column storing the item title.
available_items = [
    (int(row["ItemID"]), row["If you liked"])
    for _, row in df_collab.iterrows()
]
available_items = sorted(available_items, key=lambda x: x[0])

# ----------------------------
# Fix the Dummy Data for Reproducibility
# ----------------------------
# Set a fixed seed to ensure reproducibility
np.random.seed(42)
num_items = len(available_items)
feature_dim = 5  # Example feature dimension (replace with your real features)
X = np.random.rand(num_items, feature_dim)
# Precompute the cosine similarity matrix using the fixed dummy data
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(X)

# ----------------------------
# Recommendation Functions
# ----------------------------
def get_recommendations(item_id, sim_matrix, n=5, messages=True):
    """
    Uses a precomputed similarity matrix (e.g. cosine similarity) and returns a dict 
    of recommended item indices and scores.
    """
    if item_id >= sim_matrix.shape[0]:
        if messages:
            print(f"Item {item_id} is not in the similarity matrix provided.")
        return {}
    sim_scores = list(enumerate(sim_matrix[item_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Skip the item itself (index 0) and then take the next n items
    top_similar = sim_scores[1:n+1]
    rec_dict = {i[0]: i[1] for i in top_similar}
    if messages:
        print("get_recommendations: Recommended item indices:", list(rec_dict.keys()))
        print("get_recommendations: Similarity scores:", list(rec_dict.values()))
    return rec_dict

def recommend(itemId, X, item_mapper, item_inv_mapper, k=5, metric='cosine', messages=True):
    """
    Uses NearestNeighbors to return a list of recommended item IDs and their distances.
    """
    from sklearn.neighbors import NearestNeighbors
    rec_ids = []
    # Map the given itemId to the corresponding index in X.
    item = item_mapper[itemId]
    item_vector = X[item]
    knn = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric).fit(X)
    rec = knn.kneighbors(item_vector.reshape(1, -1), return_distance=True)
    rec_indices = rec[1][0]
    rec_distances = rec[0][0]
    # Remove the first element (the distance to itself)
    rec_distances = np.delete(rec_distances, 0)
    for i in range(1, knn.n_neighbors):
        rec_ids.append(item_inv_mapper[rec_indices[i]])
    if messages:
        print("recommend: Recommended item indices:", rec_indices)
        print("recommend: Recommended item IDs:", rec_ids)
        print("recommend: Similarity (distance) scores:", rec_distances)
    return rec_ids, rec_distances

# ----------------------------
# Mapping Dictionaries (assuming CSV ItemIDs map directly to matrix indices)
# ----------------------------
item_mapper = {item_id: item_id for item_id, _ in available_items}
item_inv_mapper = {item_id: item_id for item_id, _ in available_items}

# ----------------------------
# Flask Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            item_id = int(request.form.get("item_id").strip())
        except (ValueError, AttributeError):
            error = "Invalid item ID provided. Please select a valid item."
            return render_template("index.html", available_items=available_items, error=error)
        
        # Get Collaborative Filtering Recommendations from the functions
        rec_dict = get_recommendations(item_id, cosine_sim, n=5, messages=False)
        collab_recs = []
        for rec_item, score in rec_dict.items():
            # Look up the title from df_collab.
            title = df_collab.loc[df_collab['ItemID'] == rec_item, "If you liked"].iloc[0]
            collab_recs.append((title, round(score, 3)))
        
        # Get Content Filtering Recommendations from the functions using NearestNeighbors
        rec_ids, rec_distances = recommend(item_id, X, item_mapper, item_inv_mapper, k=5, messages=False)
        content_recs = []
        for rec_item, score in zip(rec_ids, rec_distances):
            title = df_collab.loc[df_collab['ItemID'] == rec_item, "If you liked"].iloc[0]
            content_recs.append((title, round(score, 3)))
        
        # For comparison, read the recommendations from the CSV for content filtering.
        row_content = df_content[df_content["ItemID"] == item_id]
        if not row_content.empty:
            # Assume content filtering CSV has recommendations starting with "Recommendation"
            content_cols = [col for col in df_content.columns if col.startswith("Recommendation")]
            csv_content_recs = row_content[content_cols].iloc[0].tolist() if content_cols else []
        else:
            csv_content_recs = []
        
        # Render the results; note that you'll see two sets:
        # one from the functions (content_recs) and one from the CSV (csv_content_recs)
        selected_title = df_collab.loc[df_collab['ItemID'] == item_id, "If you liked"].iloc[0]
        return render_template("result.html",
                               selected_item=selected_title,
                               collab_recs=collab_recs,
                               content_recs=content_recs,
                               csv_content_recs=csv_content_recs)
    else:
        return render_template("index.html", available_items=available_items)

if __name__ == "__main__":
    app.run(debug=True)