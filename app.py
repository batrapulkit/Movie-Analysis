import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (if required)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess the text (movie titles)
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

# Load the dataset
@st.cache
def load_data():
    # Replace with the correct path to your dataset
    df = pd.read_csv('movies.csv')
    return df

df = load_data()

# Display dataset info
st.title("Movie Data Analysis")
st.write("### Dataset Overview")
st.write(df.head())

# Basic Information
if st.checkbox("Show Basic Info"):
    st.write("### Basic Info")
    st.write(df.info())

# EDA section
st.write("### Exploratory Data Analysis")

# Plot distribution of movie ratings
if st.checkbox("Show Rating Distribution"):
    st.write("#### Rating Distribution (Histogram)")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating'], bins=20, kde=True)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    st.pyplot()

# Plot count of rating categories
if st.checkbox("Show Rating Category Distribution"):
    st.write("#### Rating Category Distribution")
    df['rating_category'] = df['rating'].apply(lambda x: 'Good' if x > 7 else ('Neutral' if 4.5 <= x <= 6.9 else 'Bad'))
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating_category', data=df, palette='Set2')
    plt.title('Distribution of Rating Categories')
    plt.xlabel('Rating Category')
    plt.ylabel('Count')
    st.pyplot()

# Missing Data Heatmap
if st.checkbox("Show Missing Data Heatmap"):
    st.write("#### Missing Data Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    st.pyplot()

# Word Cloud
st.write("### Word Cloud of Movie Titles")

if st.checkbox("Show Word Cloud"):
    # Preprocess the movie titles to generate the word cloud
    text = ' '.join(df['title'].dropna())  # Join all titles into one large string
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Display word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    plt.title('Word Cloud of Movie Titles', fontsize=15)
    st.pyplot()

# Display Top 10 movies by rating
if st.checkbox("Show Top 10 Movies by Rating"):
    st.write("#### Top 10 Movies by Rating")
    top_movies = df[['title', 'rating']].sort_values(by='rating', ascending=False).head(10)
    st.write(top_movies)

# Option to download the processed dataset (if needed)
@st.cache
def preprocess_dataset():
    df['processed_title'] = df['title'].apply(preprocess_text)
    return df

processed_df = preprocess_dataset()

if st.button('Download Processed Dataset'):
    processed_df.to_csv('processed_movies.csv', index=False)
    st.write("Processed dataset is ready for download!")
