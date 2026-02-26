import pandas as pd
import numpy as np
import ast
from collections import defaultdict
import pickle
from sklearn.preprocessing import StandardScaler

def load_artists(path):
    df_artists = pd.read_csv(path)
    return df_artists

def load_tracks(path):
    df_tracks = pd.read_csv(path)
    return df_tracks

def safe_eval(val):
    if isinstance(val, str) and val.startswith('['):
        try:
            return ast.literal_eval(val)
        except:
            return []
    return val

def string_to_list(df_w_songs, df_w_genres):
    df_w_songs['artists'] = df_w_songs['artists'].apply(safe_eval)
    df_w_songs['id_artists'] = df_w_songs['id_artists'].apply(safe_eval)
    df_w_genres['genres'] = df_w_genres['genres'].apply(safe_eval)

    return df_w_songs, df_w_genres

def remove_null(df_tracks, df_artists):
    df_songs = df_tracks.dropna(subset=['name'])
    df_artists = df_artists.dropna(subset=['name'])
    df_artists['followers'] = df_artists['followers'].fillna(0)
    return df_songs, df_artists

def build_genre_dict(df_artists):
    return dict(zip(df_artists['id'], df_artists['genres']))

def map_genres(id_artists_list, genre_dict):
    genres = []
    for artist_id in id_artists_list:
        artist_genres = genre_dict.get(artist_id, [])
        genres.extend(artist_genres)
    return list(set(genres))

def build_inverted_index(df_tracks):
    exploded = df_tracks['genres'].explode()
    inverted_index = exploded.groupby(exploded).groups
    return {genre: list(indices) for genre, indices in inverted_index.items() if pd.notna(genre)}

def scale_audio_features(df_tracks, features):
    scaler = StandardScaler()
    df_tracks[features] = scaler.fit_transform(df_tracks[features])
    return df_tracks


if __name__ == '__main__':

    print("Starting...")
    df_artists = load_artists('../data/raw/artists.csv')
    df_tracks = load_tracks('../data/raw/tracks.csv')

    print("Removing null values and converting string to list...")
    df_tracks, df_artists = remove_null(df_tracks, df_artists)
    df_tracks, df_artists = string_to_list(df_tracks, df_artists)

    df_tracks = df_tracks.reset_index(drop=True)

    print("Mapping genres...")
    g_dict = build_genre_dict(df_artists)
    df_tracks['genres'] = df_tracks['id_artists'].apply(lambda x: map_genres(x, g_dict))

    print("Making audio metadata matrix...")
    features = ["danceability",
                "energy",
                "loudness",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo"]

    audio_metadata = scale_audio_features(df_tracks, features)
    genres_index = build_inverted_index(df_tracks)

    print("Saving files...")
    audio_metadata.to_parquet('../data/processed/df_tracks.parquet')
    with open('../data/processed/genres_index.pkl', 'wb') as outfile:
        pickle.dump(genres_index, outfile)

    print("Files saved!")

