import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the association rules model from the full dataset
with open("model_full.pickle", "rb") as f:
    rules = pickle.load(f)

def get_recommendations(user_songs, rules, top_n=5):
    """
    Generate song recommendations based on a user's favorite songs.
    
    Parameters:
        user_songs (list): List of songs provided by the user.
        rules (DataFrame): Association rules generated from the dataset.
        top_n (int): Number of top recommendations to return.
        
    Returns:
        List of recommended songs.
    """
    recommendations = {}
    # Iterate over each rule in the rules DataFrame
    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        # If the rule's antecedents are all in the user's song list,
        # then consider the consequents for recommendation
        if antecedents.issubset(set(user_songs)):
            for song in consequents:
                if song not in user_songs:
                    recommendations.setdefault(song, []).append(rule['confidence'])
                    
    # Average the confidence scores for each recommended song
    rec_confidences = {song: sum(conf_list) / len(conf_list) for song, conf_list in recommendations.items()}
    # Sort recommendations by average confidence (highest first)
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
    # Run the Flask app on all interfaces on port 5000
    app.run(host="0.0.0.0", port=5000)
