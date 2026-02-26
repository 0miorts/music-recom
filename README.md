# Music Recommendation System

This repository contains the implementation of music recommendation system using the dataset from [Kaggle](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks).
The system recommends the songs based on genre similarity and audio's metadata.

## Build with
* Python 3.14
* Streamlit
* Spotipy (Spotify API Python Library)
* Pandas, NumPy, Scikit-learn

## Project structure
* `src/` - Main application, data processing and recommendation algorithms. To run processing.py you have to download Kaggle dataset.
* `data/` - processed folder contains stored data after data processing. To run processing.py put .csv files from dataset to data/raw/.
* `notebook/` - Data analysis (checking null values etc.)

```markdown
## 🚀 Run

Create `.streamlit/secrets.toml` file and add your credentials:

`client_id = "YOUR_SPOTIFY_CLIENT_ID"`

`client_secret = "YOUR_SPOTIFY_CLIENT_SECRET"`

Then run the app with:
```bash
streamlit run src/app.py
