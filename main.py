import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import copy
from sklearn.preprocessing import StandardScaler

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

mean_std_dict = {
 'danceability': {'mean': 0.6269789821124379, 'std': 0.16928329705416426},
 'energy': {'mean': 0.6111509595001064, 'std': 0.22867511151405742},
 'key': {'mean': 5.301559838160136, 'std': 3.601365222814598},
 'loudness': {'mean': -8.413418041950585, 'std': 4.773060044962174},
 'speechiness': {'mean': 0.11820605036201066, 'std': 0.12779958795305896},
 'acousticsness': {'mean': 0.294434796439521, 'std': 0.3048065194948211},
 'instrumentalness': {'mean': 0.15021815809279174, 'std': 0.3067623266279394},
 'liveliness': {'mean': 0.17988611850510877, 'std': 0.1454453882431879},
 'valence': {'mean': 0.45340037354663404, 'std': 0.24082727978555382},
 'tempo': {'mean': 121.6263309199322, 'std': 29.16301178801898},
 'duration_ms': {'mean': 207722.33698892675, 'std': 76552.8000824765},
 'age': {'mean': 528.288224020443, 'std': 26.309181841051196}
}

# Load the model, use it with loaded_model
loaded_model = joblib.load('knn_model.sav')

# Load the API, use it with loaded_api
loaded_api = SpotifyAPI('831cc784a86e40f7a94913a7760911c1', '9ec69ad406ef4de69d0c52b0becf9eb8')


def validate_input(song_features):
    # this should return ALL the missing/invalid features and not just return early.
    # for feature in required_features:
    #     if feature not in song_features:
    #         return f'{feature} requires a value'
    #
    #     # have to add validation to makes sure each value is correct.
    #     if 0 > song_features[feature] or song_features[feature] > 100:
    #         return f'{feature} value should be between 0 - 100'
    return False


def format_input(song_features):

    for key in song_features.keys():
        song_features[key] = (float(song_features[key]) - mean_std_dict[key.lower()]['mean'])/mean_std_dict[key.lower()]['std']

    df_song_features = pd.DataFrame({k: [v] for k, v in  song_features.items()})

    return df_song_features


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

    input = format_input(song_features)

    result = loaded_model.predict_proba(input)
    result = str(result[0][1])
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
