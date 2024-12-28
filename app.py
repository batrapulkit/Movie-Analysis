import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from scipy.sparse import hstack

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Load the dataset
df = pd.read_csv('movies.csv')

# Preprocessing functions (same as in your script)
def categorize_rating(rating):
    if rating > 7:
        return 'Good'
    elif 4.5 <= rating <= 6.9:
        return 'Neutral'
    else:
        return 'Bad'

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Apply text preprocessing to the 'title' column
df['processed_text'] = df['title'].apply(preprocess_text)

# Feature extraction: TF-IDF for the movie title and adding the 'ratings' as a feature
vectorizer = TfidfVectorizer(max_features=1000)
X_title = vectorizer.fit_transform(df['processed_text'])
X_ratings = np.array(df['rating']).reshape(-1, 1)
X = hstack([X_title, X_ratings])

# Labels (rating category)
y = df['rating_category']

# Load the trained model (assuming the model is already saved as 'sentiment_model.pkl')
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit interface
st.title('Movie Rating Category Prediction')

# Input section
movie_title = st.text_input('Enter a Movie Title', '')

# Prediction logic
if movie_title:
    def predict_rating_category_from_dataset(title, df):
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

    predicted_category = predict_rating_category_from_dataset(movie_title, df)
    st.write(f"Predicted category for '{movie_title}': {predicted_category}")

# Model-based prediction (optional)
st.subheader('Or use the model to predict based on title and rating')

input_title = st.text_input('Movie Title for Model Prediction', '')
input_rating = st.slider('Movie Rating', 0.0, 10.0, 5.0, 0.1)

if input_title and input_rating:
    # Preprocess the title
    processed_title = preprocess_text(input_title)
    title_vector = vectorizer.transform([processed_title])
    
    # Combine with the rating feature
    X_input = hstack([title_vector, np.array([[input_rating]])])
    
    # Predict with the model
    model_prediction = model.predict(X_input)[0]
    st.write(f"Model predicts the rating category as: {model_prediction}")
