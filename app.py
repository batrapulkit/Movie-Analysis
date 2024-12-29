import streamlit as st
import pandas as pd
import pickle
import requests
from tabulate import tabulate

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
            return "Movie not found"
    else:
        return "API Error"

def fetch_tmdb_movie_details_by_id(movie_id):
    api_key = 'da80b7c25c785e5cb5e5bc96d3f1e213'
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {'api_key': api_key}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        movie_data = response.json()
        poster_url = f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path', '')}" if movie_data.get('poster_path') else 'N/A'
        return {
            'title': movie_data.get('title', 'N/A'),
            'release_date': movie_data.get('release_date', 'N/A'),
            'overview': movie_data.get('overview', 'N/A'),
            'runtime': movie_data.get('runtime', 'N/A'),
            'platforms': fetch_dynamic_platforms(movie_data.get('title', 'N/A')),
            'poster': poster_url
        }
    else:
        return "API Error"

# OMDb API integration
def fetch_omdb_movie_details(movie_name):
    api_key = 'ca972f5'  # Replace with your OMDb API key
    base_url = "http://www.omdbapi.com/"
    params = {'apikey': api_key, 't': movie_name}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == "True":
            poster_url = data.get('Poster', 'N/A')
            return {
                'title': data.get('Title', 'N/A'),
                'year': data.get('Year', 'N/A'),
                'plot': data.get('Plot', 'N/A'),
                'actors': data.get('Actors', 'N/A'),
                'imdb_rating': data.get('imdbRating', 'N/A'),
                'runtime': data.get('Runtime', 'N/A'),
                'poster': poster_url
            }
        else:
            return "Movie not found"
    else:
        return "API Error"

# Function to fetch movie details from both APIs and display in a table format
def fetch_movie_details(movie_name):
    tmdb_details = fetch_tmdb_movie_details(movie_name)
    omdb_details = fetch_omdb_movie_details(movie_name)

    return tmdb_details, omdb_details

# Streamlit interface
st.title("Movie Rating Prediction and Details")

movie_title = st.text_input("Enter Movie Title")

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
                return "Invalid rating data"
        else:
            return "Movie not found in dataset"

    # Display rating prediction
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
    st.subheader(f"Predicted category for '{movie_title}':")
    st.write(f"**{predicted_category}**")

    # Fetch movie details
    tmdb_details, omdb_details = fetch_movie_details(movie_title)

    # Display TMDb details in a collapsible section
    with st.expander("TMDb Details", expanded=True):
        if tmdb_details != "API Error" and tmdb_details != "Movie not found":
            col1, col2 = st.columns([3, 2])
            with col1:
                if tmdb_details['poster'] != 'N/A':
                    st.image(tmdb_details['poster'], caption=f"Poster of {tmdb_details['title']}", use_container_width=True)
            with col2:
                st.write(f"**Title:** {tmdb_details.get('title', 'N/A')}")
                st.write(f"**Release Date:** {tmdb_details.get('release_date', 'N/A')}")
                st.write(f"**Overview:** {tmdb_details.get('overview', 'N/A')}")
                st.write(f"**Runtime:** {tmdb_details.get('runtime', 'N/A')} minutes")
                st.write(f"**Available on:** {', '.join(tmdb_details.get('platforms', []))}")
        else:
            st.write("No TMDb details found")

    # Display OMDb details in a collapsible section
    with st.expander("OMDb Details", expanded=True):
        if omdb_details != "API Error" and omdb_details != "Movie not found":
            col1, col2 = st.columns([3, 2])
            with col1:
                if omdb_details['poster'] != 'N/A':
                    st.image(omdb_details['poster'], caption=f"Poster of {omdb_details['title']}", use_container_width=True)
            with col2:
                st.write(f"**Title:** {omdb_details.get('title', 'N/A')}")
                st.write(f"**Year:** {omdb_details.get('year', 'N/A')}")
                st.write(f"**Plot:** {omdb_details.get('plot', 'N/A')}")
                st.write(f"**Actors:** {omdb_details.get('actors', 'N/A')}")
                st.write(f"**IMDb Rating:** {omdb_details.get('imdb_rating', 'N/A')}")
                st.write(f"**Runtime:** {omdb_details.get('runtime', 'N/A')}")
        else:
            st.write("No OMDb details found")
