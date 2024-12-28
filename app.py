import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import nltk

# Download NLTK resources explicitly to avoid the punkt_tab issue
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')  # Ensure 'punkt' tokenizer is available
    except LookupError:
        nltk.download('punkt')  # Download 'punkt' if not available
    try:
        nltk.data.find('corpora/stopwords')  # Ensure 'stopwords' are available
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')  # Ensure 'wordnet' is available for lemmatization
    except LookupError:
        nltk.download('wordnet')

# Call the function to ensure necessary data is available
ensure_nltk_data()

# Load the pre-trained model (saved as 'sentiment_model.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'sentiment_model.pkl')
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the movie dataset
df = pd.read_csv('movies.csv')

# Preprocessing function for the titles
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())  # Tokenization using 'punkt'
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Function to categorize rating
def categorize_rating(rating):
    if rating > 7:
        return 'Good'
    elif 4.5 <= rating <= 6.9:
        return 'Neutral'
    else:
        return 'Bad'

# Apply the function to the 'rating' column to create a new 'rating_category' column
df['rating_category'] = df['rating'].apply(categorize_rating)

# Function to predict rating category for a new movie title
def predict_rating_category_from_dataset(title, df, model):
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        # Preprocess the title and extract features using a TF-IDF Vectorizer
        processed_title = preprocess_text(movie_data.iloc[0]['title'])
        
        # Initialize and fit the vectorizer on the movie dataset (fit on the entire dataset or a subset)
        vectorizer = TfidfVectorizer(max_features=1000)
        X_title = vectorizer.fit_transform(df['title'].apply(preprocess_text))  # Apply preprocessing on the entire dataset

        # Transform the new title (the one entered by the user)
        title_vector = vectorizer.transform([processed_title])
        
        # Get the rating for the movie
        rating = pd.to_numeric(movie_data.iloc[0]['rating'], errors='coerce')
        
        # Combine the rating with the processed title features
        X_new = hstack([title_vector, np.array([[rating]])])
        
        # Predict the category using the trained model
        predicted_category = model.predict(X_new)
        return predicted_category[0]
    else:
        return "Movie not found in dataset"

# Streamlit interface
st.title("Movie Rating Classification")
st.write("This app classifies movies into categories based on their rating (Good, Neutral, or Bad).")

# Sidebar: Upload CSV or use built-in data
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Dataset uploaded successfully. Showing first few rows:")
    st.write(df.head())
else:
    st.write(f"Dataset loaded from default: {df.shape[0]} rows and {df.shape[1]} columns")
    st.write(df.head())

# Prediction input for movie title
movie_title = st.text_input("Enter Movie Title for Rating Prediction:")
if movie_title:
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")

# Model Evaluation Section
if st.checkbox("Show Model Evaluation (Classification Report)"):
    df['processed_text'] = df['title'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_title = vectorizer.fit_transform(df['processed_text'])
    X_ratings = np.array(df['rating']).reshape(-1, 1)
    X = hstack([X_title, X_ratings])
    y = df['rating_category']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display classification report
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

# Footer
st.write("---")
st.write("Developed by Your Name")
st.write("Streamlit Movie Rating Classification App")
