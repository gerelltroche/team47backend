from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

from SpotifyAPI import SpotifyAPI

required_features = [
    'Danceability',
    'Energy',
    'Key',
    'Loudness',
    'Speechiness',
    'Acousticsness',
    'Instrumentalness',
    'Liveliness',
    'Valence',
    'Tempo',
    'Duration_ms',
    'Age'
]


# Load the model, use it with loaded_model
loaded_model = joblib.load('knn_model.sav')


# Load the API, use it with loaded_api
loaded_api = SpotifyAPI('831cc784a86e40f7a94913a7760911c1', '9ec69ad406ef4de69d0c52b0becf9eb8')


def validate_input(song_features):
    # this should return ALL the missing/invalid features and not just return early.
    for feature in required_features:
        if feature not in song_features:
            return f'{feature} requires a value'

        # have to add validation to makes sure each value is correct.
        # if 0 > song_features[feature] or song_features[feature] > 100:
        #     return f'{feature} value should be between 0 - 100'


def format_input(song_features):
    # Create 1-line DF
    df = pd.DataFrame({k: [v] for k, v in song_features.items()})

    # format with scaler
    formatted = StandardScaler().fit_transform(df)

    return formatted


# Start the server listening.. PORT 5000 by default.
app = Flask(__name__)
CORS(app)


# Note: Do escape(stuff) on any unsafe stuff to avoid SQL injection

@app.route("/")
def index():
    return "<p>This is the backend endpoint. do a GET request to /predict with the song features </p>"


@app.route("/predict", methods=['GET'])
def predict():

    song_features = request.args.to_dict()
    error = validate_input(song_features)

    if error:
        return error

    result = loaded_model.predict_proba(format_input(song_features))
    result = str(result)
    print(result)
    return result


@app.route("/autocomplete/<id>", methods=['GET'])
def autocomplete(id):
    response = loaded_api.topFiveTracks(id)

    return response


@app.route('/singlelookup/<id>', methods=['GET'])
def singlelookup(id):
    response = loaded_api.audiofeatSingle(id)
    print(response)
    return response
