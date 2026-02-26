from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st

def spotify_auth():
    try:
        client_id = st.secrets["SPOTIPY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]
    except KeyError:
        st.error("Keys not found")
        st.stop()
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth_manager)

def genre_similarity(target_idx, df, genres_dict):
    target_genres = df.loc[target_idx, 'genres']
    if target_genres is None or len(target_genres) == 0:
        return np.full(len(df), 0.5)
    genre_bonus = np.zeros(len(df))
    for g in genres_dict:
        if g in target_genres:
            genre_bonus[genres_dict[g]] += 1
    return genre_bonus / len(target_genres)

def get_recommendations(target_idx, df, genres_dict, features, n_recommendations = 5, alpha_weight = 0.7):
    target_audio = df.loc[[target_idx], features].values
    audio_sim = cosine_similarity(target_audio, df[features].values).flatten()

    genre_sim = genre_similarity(target_idx, df, genres_dict)

    final_sim = (alpha_weight * audio_sim + (1 - alpha_weight) * genre_sim)
    scores = final_sim.copy()
    scores[scores > 0.99] = 0
    idx = scores.argsort()[-n_recommendations:][::-1]
    return idx, final_sim[idx]


def get_image(target_idx, sp, df):
    try:
        spotify_id = df.loc[target_idx, 'id']
        result = sp.track(spotify_id)
        return result['album']['images'][0]['url']
    except:
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/No-Image-Placeholder.svg/500px-No-Image-Placeholder.svg.png"