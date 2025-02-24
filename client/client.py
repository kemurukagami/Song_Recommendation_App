#!/usr/bin/env python3
import sys
import json
import requests

def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <IP> <song1> [<song2> ...]")
        sys.exit(1)

    # Collect the list of songs from the command-line arguments
    IP = sys.argv[1]
    songs = sys.argv[2:]
    payload = {"songs": songs}

    # For isaac, the Flask server runs on port 52008.
    url = f"http://{IP}:52008/api/recommender"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses (4XX, 5XX)
        data = response.json()
        print("Recommendations:")
        for song in data.get("songs", []):
            print("  -", song)
        print("Version:", data.get("version", "N/A"))
        print("Model Date:", data.get("model_date", "N/A"))
    except requests.exceptions.RequestException as e:
        print("Error contacting the server:", e)

if __name__ == "__main__":
    main()
