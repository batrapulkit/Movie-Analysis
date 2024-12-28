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

# NLTK Data Downloads (ensure these are available in Streamlit Cloud)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model and vectorizer from the saved files
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the movie dataset
df = pd.read_csv('movies.csv')

# Preprocessing function
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
def predict_rating_category_from_dataset(title, df, model, vectorizer):
    # Check if the movie title exists in the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]
    
    if not movie_data.empty:
        # Preprocess the title and extract features using the vectorizer
        processed_title = preprocess_text(movie_data.iloc[0]['title'])
        title_vector = vectorizer.transform([processed_title])
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
    predicted_category = predict_rating_category_from_dataset(movie_title, df, model, vectorizer)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")

# Model Evaluation Section
if st.checkbox("Show Model Evaluation (Classification Report)"):
    # Extract features and labels
    df['processed_text'] = df['title'].apply(preprocess_text)
    X_title = vectorizer.transform(df['processed_text'])
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
