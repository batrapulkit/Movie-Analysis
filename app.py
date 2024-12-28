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

# Streamlit interface
st.title("Movie Rating Prediction")

movie_title = st.text_input("Enter Movie Title")

if movie_title:
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")
