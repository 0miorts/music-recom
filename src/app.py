import pandas as pd
import pickle
import re
from streamlit import text_input
from engine import get_recommendations, spotify_auth, get_image
import streamlit as st
import os
import numpy as np

features = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKS_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "df_tracks.parquet")
GENRES_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "genres_index.pkl")


@st.cache_resource
def load_data():
    if not os.path.exists(TRACKS_PATH):
        st.error(f"Nie znaleziono pliku: {TRACKS_PATH}")
        st.stop()
    df = pd.read_parquet(TRACKS_PATH)
    df['search_index'] = (
            df['name'].astype(str).str.lower() +
            " " +
            df['artists'].astype(str).str.lower()
    )
    with open(GENRES_PATH, 'rb') as f:
        g_dict = pickle.load(f)

    return df, g_dict


df, genres_dict = load_data()

df, genres_dict = load_data()

sp = spotify_auth()
st.title('Spotify Track Recommendation 🎵')

user_input = st.text_input("Search for song or artist", placeholder='Young Thug').strip()
if user_input:
    pattern = rf'\b{re.escape(user_input)}\b'
    mask = df['search_index'].str.contains(pattern, case=False, na=False, regex=True)
    results = df[mask].sort_values(by="popularity", ascending=False).head(15)
    if not results.empty:
        st.write('Choose the exact song:')
        options = results.apply(lambda x: f"{x['name']} - {x['artists']}", axis=1).tolist()
        selection = st.selectbox("Select song: ", options)
        if selection:
            target_idx = results.index[options.index(selection)]
            st.success(f"Selected: {selection}")
            st.divider()
            with st.spinner('Processing...'):
                rec_indices, scores = get_recommendations(target_idx, df, genres_dict, features)

            st.subheader("Recommended for you:")

            for i, idx in enumerate(rec_indices):
                col1, col2 = st.columns([1, 4])
                img_url = get_image(idx, sp, df)

                track_name = df.loc[idx, 'name']
                artist_name = df.loc[idx, 'artists']
                with col1:
                    st.image(img_url, width=100)

                with col2:
                    if isinstance(artist_name, (list, np.ndarray)):
                        clean_artist = ", ".join(artist_name)
                    else:
                        clean_artist = str(artist_name).strip("[]'").replace("' '", ", ").replace("'", "")
                    st.write(f"**{track_name}**")
                    st.write(f"by {clean_artist}")


