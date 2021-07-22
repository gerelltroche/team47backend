from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler

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
    'Duration_ms'
]

loaded_model = joblib.load('knn_model.sav')


# Do escape() on any unsafe stuff

def validate_input(song_features):
    for feature in required_features:
        if feature not in song_features:
            return f'{feature} requires a value'

        # Do this style validation for each input
        if 0 > song_features[feature] or song_features[feature] > 100:
            return f'{feature} value should be between 0 - 100'


def format_input(song_features):
    # do cool stuff


    # create the age variable in days as an integer from release date.

    # map it all to a dictionary or DF if it doesn't work.

    song = thedictionary

    scaled_song = StandardScaler().fit_transform(song)

    loaded_model.predict_proba(scaled_song)

    return None


app = Flask(__name__)


@app.route("/")
def index():
    return "<p>This is the backend endpoint. do a GET request to /predict with the song features </p>"


@app.route("/predict", methods=['GET'])
def predict():
    song_features = request.args.to_dict()
    error = validate_input(song_features)

    if error:
        return error

    result = loaded_model.predict_proba(format_input(song_features)) # <--- how do i format the X_test and Y_test?

    return song_features
