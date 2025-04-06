import os
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
# Assume the collaborative CSV has an "If you liked" column storing item titles.
available_items = [
    (int(row["ItemID"]), row["If you liked"])
    for _, row in df_collab.iterrows()
]
available_items = sorted(available_items, key=lambda x: x[0])

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
        
        # ----------------------------
        # Get Collaborative Filtering Recommendations from CSV
        # ----------------------------
        row_collab = df_collab[df_collab["ItemID"] == item_id]
        if row_collab.empty:
            error = f"ItemID {item_id} not found in Collaborative Filtering data."
            return render_template("index.html", available_items=available_items, error=error)
        # Get the title of the selected item
        selected_title = row_collab["If you liked"].iloc[0]
        # Look for columns starting with "Recommendation" (e.g., "Recommendation 1", "Recommendation 2", etc.)
        collab_cols = [col for col in df_collab.columns if col.startswith("Recommendation")]
        if collab_cols:
            collab_recs = row_collab[collab_cols].iloc[0].tolist()
        else:
            collab_recs = []
        
        # ----------------------------
        # Get Content Filtering Recommendations from CSV
        # ----------------------------
        row_content = df_content[df_content["ItemID"] == item_id]
        if not row_content.empty:
            # Try to extract recommendation columns (adjust this if your CSV uses different column names)
            content_cols = [col for col in df_content.columns if col.startswith("Recommendation")]
            if content_cols:
                content_recs = row_content[content_cols].iloc[0].tolist()
            else:
                # Alternatively, if there is a single column (for example, "If you liked" or "title") that serves as a recommendation
                content_recs = [row_content["If you liked"].iloc[0]] if "If you liked" in row_content.columns else []
        else:
            content_recs = []
        
        return render_template("result.html", selected_item=selected_title,
                               collab_recs=collab_recs, content_recs=content_recs)
    else:
        # On GET: render the homepage with a dropdown built from the available items list.
        return render_template("index.html", available_items=available_items)

if __name__ == "__main__":
    app.run(debug=True)