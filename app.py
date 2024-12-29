import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the pre-trained sentiment model and dataset
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')  # Replace with your dataset path
    return df

# Placeholder function for dynamic platform data
def fetch_dynamic_platforms(movie_name):
    platforms = {
        "Guardians of the Galaxy Vol. 2": ["Netflix", "Disney+"],
        "The Dark Knight": ["HBO Max", "Amazon Prime", "Netflix"],
        "Inception": ["Netflix", "Hulu", "Amazon Prime"]
    }
    return platforms.get(movie_name, ["Platform info not available"])

# TMDb API integration - Fetch detailed movie info for runtime and other data
def fetch_tmdb_movie_details(movie_name):
    api_key = 'da80b7c25c785e5cb5e5bc96d3f1e213'  # Replace with your TMDb API key
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {'api_key': api_key, 'query': movie_name}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            movie_id = movie.get('id')
            return fetch_tmdb_movie_details_by_id(movie_id)
        else:
            return None  # Return None when movie is not found
    else:
        return None  # Return None for API errors

def fetch_tmdb_movie_details_by_id(movie_id):
    api_key = 'da80b7c25c785e5cb5e5bc96d3f1e213'
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {'api_key': api_key}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        movie_data = response.json()
        return {
            'title': movie_data.get('title', 'N/A'),
            'release_date': movie_data.get('release_date', 'N/A'),
            'overview': movie_data.get('overview', 'N/A'),
            'runtime': movie_data.get('runtime', 'N/A'),
            'platforms': fetch_dynamic_platforms(movie_data.get('title', 'N/A')),
            'poster': f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path', '')}"
        }
    else:
        return None  # Return None for API errors

# OMDb API integration
def fetch_omdb_movie_details(movie_name):
    api_key = 'ca972f5'  # Replace with your OMDb API key
    base_url = "http://www.omdbapi.com/"
    params = {'apikey': api_key, 't': movie_name}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == "True":
            return {
                'title': data.get('Title', 'N/A'),
                'year': data.get('Year', 'N/A'),
                'plot': data.get('Plot', 'N/A'),
                'actors': data.get('Actors', 'N/A'),
                'imdb_rating': data.get('imdbRating', 'N/A'),
                'runtime': data.get('Runtime', 'N/A'),
                'poster': data.get('Poster', 'N/A')
            }
        else:
            return None  # Return None when movie is not found
    else:
        return None  # Return None for API errors

# Function to fetch movie details from both APIs and display in a table format
def fetch_movie_details(movie_name):
    tmdb_details = fetch_tmdb_movie_details(movie_name)
    omdb_details = fetch_omdb_movie_details(movie_name)

    return tmdb_details, omdb_details

# Streamlit interface
st.set_page_config(page_title="Movie Rating Prediction", page_icon=":movie_camera:", layout="wide")

# Introduction
st.title("🎬 Movie Rating Prediction & Details")
st.write(
    "Welcome to the Movie Rating Prediction app! Enter a movie title below to predict its rating and get detailed movie information like runtime, plot, and where to watch it."
)

# Movie Title Input without the sidebar
movie_title = st.text_input("Enter Movie Title", key="movie_title", placeholder="e.g., Inception")

# Main Content: If movie title is entered, automatically show details
if movie_title:
    # Load dataset and model
    df = load_data()
    model = load_model()

    # Movie rating prediction
    def preprocess_text(text):
        return text.lower()

    def predict_rating_category_from_dataset(title, df, model):
        processed_title = preprocess_text(title)
        movie_data = df[df['title'].str.contains(title, case=False, na=False)]
        
        if not movie_data.empty:
            rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')
            if not pd.isna(rating):
                if rating >= 7:
                    return "Good"
                elif 5 <= rating < 7:
                    return "Neutral"
                else:
                    return "Bad"
            else:
                return None  # Return None if rating data is not available
        else:
            return None  # Return None when movie is not found in dataset

    # Get the predicted rating category
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)

    # Only display predicted rating if valid
    if predicted_category:
        st.subheader(f"Predicted Rating for '{movie_title}': {predicted_category}")

    # Fetch movie details
    tmdb_details, omdb_details = fetch_movie_details(movie_title)

    # Only display movie details if found
    if tmdb_details:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("TMDb Movie Details")
            st.image(tmdb_details['poster'], caption=f"Poster of {tmdb_details['title']}", width=200)
            st.markdown(f"**Title:** {tmdb_details.get('title', 'N/A')}")
            st.markdown(f"**Release Date:** {tmdb_details.get('release_date', 'N/A')}")
            st.markdown(f"**Overview:** {tmdb_details.get('overview', 'N/A')}")
            st.markdown(f"**Runtime:** {tmdb_details.get('runtime', 'N/A')} minutes")
            st.markdown(f"**Available on:** {', '.join(tmdb_details.get('platforms', []))}")
    
    if omdb_details:
        with col2:
            st.subheader("OMDb Movie Details")
            st.image(omdb_details['poster'], caption=f"Poster of {omdb_details['title']}", width=200)
            st.markdown(f"**Title:** {omdb_details.get('title', 'N/A')}")
            st.markdown(f"**Year:** {omdb_details.get('year', 'N/A')}")
            st.markdown(f"**Plot:** {omdb_details.get('plot', 'N/A')}")
            st.markdown(f"**Actors:** {omdb_details.get('actors', 'N/A')}")
            st.markdown(f"**IMDb Rating:** {omdb_details.get('imdb_rating', 'N/A')}")
            st.markdown(f"**Runtime:** {omdb_details.get('runtime', 'N/A')}")
