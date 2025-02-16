import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the association rules model from the full dataset
with open("model_full.pickle", "rb") as f:
    rules = pickle.load(f)

def get_recommendations(user_songs, rules, top_n=5):
    recommendations = {}
    user_set = set(user_songs)
    print("User songs:", user_set)  # Debug print

    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        # Debug: print rule details
        print(f"Evaluating rule: if {antecedents} then {consequents}")
        if antecedents.issubset(user_set):
            print("Match found for rule:", rule)
            for song in consequents:
                if song not in user_set:
                    recommendations.setdefault(song, []).append(rule['confidence'])
                    
    rec_confidences = {song: sum(conf_list) / len(conf_list) for song, conf_list in recommendations.items()}
    sorted_recs = sorted(rec_confidences.items(), key=lambda x: x[1], reverse=True)
    
    return [song for song, conf in sorted_recs][:top_n]


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    user_songs = data.get("songs", [])
    
    if not user_songs:
        return jsonify({"error": "No songs provided in the request."}), 400
    
    # Get recommendations based on the user's input songs
    recommendations = get_recommendations(user_songs, rules)
    
    response = {
        "songs": recommendations,
        "version": "1.0",
        "model_date": "2023-04-20"  # Update with the actual model update date as needed
    }
    
    return jsonify(response)

if __name__ == '__main__':
    # Run the Flask app on all interfaces on port 52008
    app.run(host="0.0.0.0", port=52008)
