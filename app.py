import streamlit as st
import joblib
import pandas as pd

# Load the trained model (Make sure the model is saved as a .pkl file)
model = joblib.load('sentiment_model.pkl')  # Assuming your model is saved as sentiment_model.pkl
# Optionally, load any other files (like encoders, scalers) if used
# encoder = joblib.load('encoder.pkl')  # if you used one for encoding categorical data

# Helper function to predict rating category
def predict_rating_category(title, model, df):
    # Check if the movie exists in the dataset
    movie_data = df[df['title'].str.contains(title, case=False, na=False)]

    if not movie_data.empty:
        rating = movie_data.iloc[0]['ratings']  # Assuming 'ratings' column exists
        # Predict the category based on the rating
        if rating >= 7:
            return "Good"
        elif 4.5 <= rating < 7:
            return "Neutral"
        else:
            return "Bad"
    else:
        return "Movie not found in the dataset."

# Title of the app
st.title('Movie Rating Category Classifier')

# Instructions for the user
st.write("""
    This app allows you to predict the category of a movie based on its rating.
    Enter the movie title, and the app will classify the movie as **Good**, **Neutral**, or **Bad** based on its rating.
""")

# Create a text input field for the movie title
movie_title = st.text_input('Enter Movie Title')

# Add a button for the user to trigger the prediction
if st.button('Predict'):
    # Load the dataset (this can be a larger dataset from CSV or a database)
    df = pd.read_csv('movies_dataset.csv')  # Assuming you have a CSV dataset of movies

    if movie_title:
        # Call the function to predict the rating category
        result = predict_rating_category(movie_title, model, df)
        st.write(f"Predicted Category for '{movie_title}': {result}")
    else:
        st.write("Please enter a movie title to get the prediction.")

# Optional: Display some data from the dataset (for context)
if st.checkbox('Show Sample Movies Dataset'):
    st.write(df.head())
