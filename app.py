import streamlit as st
import pandas as pd
import requests
from tabulate import tabulate

# Custom CSS for styling
st.markdown("""
    <style>
    .header {
        font-size: 40px;
        font-weight: bold;
        color: #FF6347;
    }
    .subtitle {
        font-size: 30px;
        font-weight: 500;
        color: #008080;
    }
    .movie-title {
        font-size: 24px;
        font-weight: 600;
        color: #2F4F4F;
    }
    .movie-details {
        font-size: 18px;
        color: #555;
    }
    .movie-card {
        border: 2px solid #FF6347;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f8f8;
    }
    .movie-overview {
        font-style: italic;
        color: #808080;
    }
    </style>
    """, unsafe_allow_html=True)

# TMDb API integration - Fetch movie details
def fetch_tmdb_movie_details(movie_name):
    api_key = 'your_tmdb_api_key'
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': api_key,
        'query': movie_name
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            movie_id = movie.get('id')  # TMDb movie ID
            return fetch_tmdb_movie_details_by_id(movie_id)
        else:
            return "Movie not found"
    else:
        return "API Error"
    
def fetch_tmdb_movie_details_by_id(movie_id):
    api_key = 'your_tmdb_api_key'
    base_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        'api_key': api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        movie_data = response.json()
        return {
            'title': movie_data.get('title'),
            'release_date': movie_data.get('release_date'),
            'overview': movie_data.get('overview'),
            'runtime': movie_data.get('runtime', 'N/A'),  # TMDb provides runtime
        }
    else:
        return "API Error"

# OMDb API integration - Fetch movie details
def fetch_omdb_movie_details(movie_name):
    api_key = 'your_omdb_api_key'
    base_url = "http://www.omdbapi.com/"
    params = {
        'apikey': api_key,
        't': movie_name
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == "True":
            return {
                'title': data.get('Title'),
                'year': data.get('Year'),
                'plot': data.get('Plot'),
                'actors': data.get('Actors'),
                'imdb_rating': data.get('imdbRating'),
                'runtime': data.get('Runtime')  # OMDb runtime
            }
        else:
            return "Movie not found"
    else:
        return "API Error"

# Function to fetch movie details from both APIs and display separately
def fetch_movie_details(movie_name):
    tmdb_details = fetch_tmdb_movie_details(movie_name)
    omdb_details = fetch_omdb_movie_details(movie_name)

    # Layout for movie details display
    if tmdb_details != "Movie not found" and omdb_details != "Movie not found":
        col1, col2 = st.columns(2)

        # TMDb movie details in column 1
        with col1:
            st.markdown(f"### TMDb Movie Details")
            st.markdown(f"**Title:** {tmdb_details['title']}")
            st.markdown(f"**Release Date:** {tmdb_details['release_date']}")
            st.markdown(f"**Runtime:** {tmdb_details['runtime']} minutes")
            st.markdown(f"**Overview:** {tmdb_details['overview'][:250]}...")

        # OMDb movie details in column 2
        with col2:
            st.markdown(f"### OMDb Movie Details")
            st.markdown(f"**Title:** {omdb_details['title']}")
            st.markdown(f"**Year:** {omdb_details['year']}")
            st.markdown(f"**IMDB Rating:** {omdb_details['imdb_rating']}")
            st.markdown(f"**Runtime:** {omdb_details['runtime']}")
            st.markdown(f"**Actors:** {omdb_details['actors']}")
            st.markdown(f"**Plot:** {omdb_details['plot'][:250]}...")

# Streamlit frontend
st.title("Movie Information Search")
movie_title = st.text_input("Enter Movie Title")

if movie_title:
    with st.spinner('Fetching movie details...'):
        fetch_movie_details(movie_title)

st.markdown("""
    <footer style="text-align:center; padding-top: 30px;">
        <p style="font-size: 14px; color: #777;">Powered by Streamlit | <a href="https://www.themoviedb.org/" target="_blank">TMDb</a> & <a href="https://www.omdbapi.com/" target="_blank">OMDb</a></p>
    </footer>
    """, unsafe_allow_html=True)
