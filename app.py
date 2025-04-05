import pandas as pd
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# ---------------------------
# Load Collaborative Filtering Data
# ---------------------------
try:
    collab_df = pd.read_csv('./data/Collaborative_Filtering.csv')
    # Remove extra spaces from header names.
    collab_df.columns = collab_df.columns.str.strip()
except FileNotFoundError:
    raise FileNotFoundError("The file 'Collaborative_Filtering.csv' was not found. Verify the file path.")

# Ensure the expected column is present and define the ItemID.
if "If you liked" not in collab_df.columns:
    raise KeyError("Collaborative_Filtering.csv must contain the column 'If you liked'.")
collab_df['ItemID'] = collab_df['If you liked'].astype(str)

# ---------------------------
# Load Content Filtering Data
# ---------------------------
try:
    content_df = pd.read_csv('./data/content_filtering.csv')
    # Remove extra spaces from header names.
    content_df.columns = content_df.columns.str.strip()
except FileNotFoundError:
    raise FileNotFoundError("The file 'content_filtering.csv' was not found. Verify the file path.")

# For content filtering, use the 'title' column as the lookup key since the CSV's index is as described.
if "title" in content_df.columns:
    content_df['ItemID'] = content_df['title'].astype(str)
else:
    # Fallback: use the first column as the identifier.
    print("Warning: 'title' column not found in content_filtering.csv; using the first column as the ItemID.")
    content_df['ItemID'] = content_df.iloc[:, 0].astype(str)

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the selected item from the dropdown.
        item_id = request.form.get('item_id')
        
        # Filter the collaborative filtering data:
        row_collab = collab_df[collab_df['ItemID'] == item_id]
        if row_collab.empty:
            error = f"Item ID '{item_id}' not found in Collaborative Filtering data."
            item_ids = sorted(collab_df['ItemID'].unique())
            return render_template('index.html', item_ids=item_ids, error=error)
        collab_recommendations = row_collab.iloc[0].to_dict()
        
        # Filter the content filtering data:
        row_content = content_df[content_df['ItemID'] == item_id]
        content_recommendations = row_content.to_dict('records') if not row_content.empty else []
        
        return render_template('result.html',
                               collab_recommendations=collab_recommendations,
                               content_recommendations=content_recommendations)
    else:
        # On GET: build a dropdown using ItemIDs from the Collaborative Filtering data.
        item_ids = sorted(collab_df['ItemID'].unique())
        return render_template('index.html', item_ids=item_ids)

if __name__ == '__main__':
    app.run(debug=True)