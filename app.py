import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from tabulate import tabulate

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained sentiment model and dataset
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')  # Replace with the correct path to your dataset
    return df

# Preprocess text (title)
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(str(text).lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Function to predict the rating category for a movie using the movie title
def predict_rating_category_from_dataset(title, df, model):
    processed_title = preprocess_text(title)

    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')  # Convert to numeric
        
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

# TMDb API integration - Fetch detailed movie info for runtime and other data
def fetch_tmdb_movie_details(movie_name):
    api_key = 'your_tmdb_api_key'  # Replace with your TMDb API key
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
            movie_id = movie.get('id')
            return fetch_tmdb_movie_details_by_id(movie_id)
        else:
            return "Movie not found"
    else:
        return "API Error"

def fetch_tmdb_movie_details_by_id(movie_id):
    api_key = 'your_tmdb_api_key'  # Replace with your TMDb API key
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
            'runtime': movie_data.get('runtime', 'N/A'),
            'platforms': fetch_dynamic_platforms(movie_data.get('title'))
        }
    else:
        return "API Error"

# OMDb API integration
def fetch_omdb_movie_details(movie_name):
    api_key = 'your_omdb_api_key'  # Replace with your OMDb API key
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
                'runtime': data.get('Runtime')
            }
        else:
            return "Movie not found"
    else:
        return "API Error"

# Placeholder function for dynamic platform data (replace with actual API call)
def fetch_dynamic_platforms(movie_name):
    platforms = {
        "Guardians of the Galaxy Vol. 2": ["Netflix", "Disney+"],
        "The Dark Knight": ["HBO Max", "Amazon Prime", "Netflix"],
        "Inception": ["Netflix", "Hulu", "Amazon Prime"]
    }
    return platforms.get(movie_name, ["Platform info not available"])

# Function to fetch movie details from both APIs and display in a table format
def fetch_movie_details(movie_name):
    tmdb_details = fetch_tmdb_movie_details(movie_name)
    omdb_details = fetch_omdb_movie_details(movie_name)

    if tmdb_details != "Movie not found" and omdb_details != "Movie not found":
        table = [
            ['Attribute', 'TMDb', 'OMDb'],
            ['Title', tmdb_details['title'], omdb_details['title']],
            ['Release Date', tmdb_details['release_date'], 'N/A'],
            ['Overview', tmdb_details['overview'], 'N/A'],
            ['Platforms', ', '.join(tmdb_details['platforms']), 'N/A'],
            ['Year', 'N/A', omdb_details['year']],
            ['Plot', 'N/A', omdb_details['plot']],
            ['Actors', 'N/A', omdb_details['actors']],
            ['IMDB Rating', 'N/A', omdb_details['imdb_rating']],
            ['Runtime', tmdb_details['runtime'], omdb_details['runtime']]
        ]
        return tabulate(table, headers="firstrow", tablefmt="grid")
    else:
        return "Movie not found in both APIs"

# Streamlit interface
st.title("Movie Rating Prediction and Information")

# Movie title input
movie_title = st.text_input("Enter Movie Title")

if movie_title:
    # Fetch movie details from APIs
    movie_details = fetch_movie_details(movie_title)
    st.write(movie_details)
    
    # Load dataset and model
    df = load_data()
    model = load_model()
    
    # Predict rating category
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")
