import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from tabulate import tabulate

# Download necessary NLTK resources (ensure this is available in your environment)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model and dataset
@st.cache_resource
def load_model():
    # Load the pre-trained sentiment model
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_data():
    # Replace with the correct path to your dataset
    df = pd.read_csv('movies.csv')
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
    # Preprocess the title (ensure the same preprocessing is applied to the input text)
    processed_title = preprocess_text(title)

    # Check if the movie title exists in the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    # If movie is found in the dataset
    if not movie_data.empty:
        # Extract the rating from the dataset and convert it to a numeric value
        rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')  # Convert to numeric
        
        # If rating is not NaN (i.e., valid rating value)
        if not pd.isna(rating):
            # Predict the category based on the rating
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
    api_key = 'da80b7c25c785e5cb5e5bc96d3f1e213'  # Replace with your TMDb API key
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
            # Fetch detailed information using the TMDb movie ID
            return fetch_tmdb_movie_details_by_id(movie_id)
        else:
            return "Movie not found"
    else:
        return "API Error"
    
def fetch_tmdb_movie_details_by_id(movie_id):
    api_key = 'da80b7c25c785e5cb5e5bc96d3f1e213'  # Replace with your TMDb API key
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
            'platforms': fetch_dynamic_platforms(movie_data.get('title'))  # Fetch dynamic platforms
        }
    else:
        return "API Error"

# OMDb API integration
def fetch_omdb_movie_details(movie_name):
    api_key = 'ca972f5'  # Replace with your OMDb API key
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

# Placeholder function for dynamic platform data (replace with actual API call)
def fetch_dynamic_platforms(movie_name):
    # Placeholder logic for fetching dynamic platform data
    platforms = {
        "Guardians of the Galaxy Vol. 2": ["Netflix", "Disney+"],
        "The Dark Knight": ["HBO Max", "Amazon Prime", "Netflix"],
        "Inception": ["Netflix", "Hulu", "Amazon Prime"]
    }
    return platforms.get(movie_name, ["Platform info not available"])

# Streamlit interface
st.title("Movie Rating Prediction and Information")

movie_title = st.text_input("Enter Movie Title")

if movie_title:
    # Load model and data
    model = load_model()
    df = load_data()
    
    # Sentiment prediction from dataset
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
    
    # Fetch movie details from TMDb and OMDb
    tmdb_details = fetch_tmdb_movie_details(movie_title)
    omdb_details = fetch_omdb_movie_details(movie_title)
    
    # Display sentiment prediction
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")
    
    # Display movie details in a table format
    if tmdb_details != "Movie not found" and omdb_details != "Movie not found":
        # Combine data and display in a tabular format
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
        st.table(table)
    else:
        st.write("Movie not found in TMDb or OMDb.")
