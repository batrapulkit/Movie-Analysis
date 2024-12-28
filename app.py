import streamlit as st
import pandas as pd
import numpy as np
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import nltk
from nltk.tokenize import word_tokenize

# Download punkt tokenizer models if they are not available
nltk.download('punkt')


# Ensure that NLTK uses local resources
nltk.data.path.append('nltk_data')  # Add the local nltk_data directory to the search path

# Load pre-trained model and necessary data
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the movie dataset for predicting movie ratings
df = pd.read_csv('movies.csv')

# Function to preprocess text for prediction
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())  # Tokenizing the text
    stop_words = set(stopwords.words('english'))  # Get stop words
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]  # Removing non-alphabetic and stopwords
    lemmatizer = WordNetLemmatizer()  # Lemmatizing the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    return ' '.join(tokens)

# Function to categorize movie ratings
def categorize_rating(rating):
    if rating > 7:
        return 'Good'
    elif 4.5 <= rating <= 6.9:
        return 'Neutral'
    else:
        return 'Bad'

# Function to predict the rating category from the dataset
def predict_rating_category_from_dataset(title, df):
    # Check if the movie title exists in the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        # Extract the rating from the dataset and convert it to a numeric value
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

# Streamlit App
def main():
    st.title("Movie Rating Prediction")
    
    # User Input for Movie Title
    movie_title = st.text_input("Enter the Movie Title:")
    
    if movie_title:
        # Preprocess the movie title for prediction
        processed_title = preprocess_text(movie_title)
        
        # Feature extraction for the movie title (TF-IDF + rating feature)
        vectorizer = TfidfVectorizer(max_features=1000)
        X_title = vectorizer.fit_transform([processed_title])
        
        # Add the rating (assuming we use a default rating value or fetch from the dataset)
        X_ratings = np.array([5.5]).reshape(-1, 1)  # Example: neutral rating value
        X = np.hstack([X_title.toarray(), X_ratings])
        
        # Predict using the trained model
        prediction = model.predict(X)
        
        # Map prediction to readable label
        result = prediction[0]
        
        st.subheader(f"Predicted Category: {result}")
    
    # Check prediction from the dataset based on title
    if st.button("Predict from Dataset"):
        result = predict_rating_category_from_dataset(movie_title, df)
        st.subheader(f"Dataset Prediction: {result}")

if __name__ == "__main__":
    main()
