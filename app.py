import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Load the pre-trained model and dataset
@st.cache_resource
def load_model():
    # Load the pre-trained model
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_data():
    # Replace with the correct path to your dataset
    df = pd.read_csv('movies.csv')
    return df

# Load model and data
model = load_model()
df = load_data()

# Function to predict the rating category for a movie using the movie title
def predict_rating_category_from_dataset(title, df, model):
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

