from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os
import time

app = Flask(__name__, template_folder="templates")
CORS(app)  # Enable CORS for frontend integration

# Check if running inside Docker (Docker sets the `container` environment variable)
IN_DOCKER = os.path.exists("/app")

MODEL_PATH = "/app/shared/model/model.pickle" if IN_DOCKER else "./model/model_full.pickle"

# Initialize global variables
model = None
MODEL_VERSION = "N/A"
MODEL_DATE = "N/A"
last_modified = None

def load_model():
    """Loads the model from file if it has changed."""
    global model, MODEL_VERSION, MODEL_DATE, last_modified
    
    if not os.path.exists(MODEL_PATH):
        model = None
        MODEL_VERSION = "N/A"
        MODEL_DATE = "N/A"
        return

    # Check the last modified timestamp
    modified_time = os.path.getmtime(MODEL_PATH)
    
    if last_modified is None or modified_time > last_modified:
        print("Reloading model...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        last_modified = modified_time
        MODEL_VERSION = "0.1"
        MODEL_DATE = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(modified_time))

# Serve a basic web-based frontend
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint to get recommendations
@app.route("/api/recommender", methods=["POST"])
def recommend():
    load_model()  # Reload model if updated
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        input_songs = set(data.get("songs", []))

        recommended_songs = set()
        for _, row in model.iterrows():
            antecedents = set(row["antecedents"])
            consequents = set(row["consequents"])
            if antecedents.issubset(input_songs):
                recommended_songs.update(consequents)

        return jsonify({
            "songs": list(recommended_songs),
            "version": MODEL_VERSION,
            "model_date": MODEL_DATE
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 52008))  # Allow custom port configuration
    app.run(host="0.0.0.0", port=port)
