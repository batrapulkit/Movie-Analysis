# Movie Analysis

This project is a comprehensive movie analysis model that predicts movie ratings and categorizes them based on their performance. The model utilizes machine learning algorithms and natural language processing (NLP) techniques to analyze movie titles, ratings, and other features to provide insights on the movies' reception.

## Project Overview

The Movie Analysis model is designed to:
- Predict movie rating categories based on movie titles and other features.
- Analyze movie reviews to categorize them as "Good," "Neutral," or "Bad" based on their ratings.

### Key Features:
- **Rating Categorization**: Categorizes movie ratings into three categories: "Good" (rating > 7), "Neutral" (rating between 4.5 and 6.9), and "Bad" (rating < 4.5).
- **Natural Language Processing (NLP)**: Uses text preprocessing techniques like tokenization, stopword removal, and lemmatization to process movie titles.
- **Machine Learning**: Trains a logistic regression model using movie titles (TF-IDF features) and ratings to predict rating categories.

## Requirements

This project requires the following libraries:

- Python 3.x
- `pandas` – For data manipulation
- `scikit-learn` – For machine learning models
- `nltk` – For natural language processing
- `numpy` – For numerical operations
- `scipy` – For sparse matrix operations

To install the dependencies, run:

```bash
pip install pandas scikit-learn nltk numpy scipy
