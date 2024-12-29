# 🎬 Movie Rating Prediction and Details 🎬

This web app allows users to predict the rating category of a movie (Good, Neutral, or Bad) based on a dataset, and fetch detailed information about the movie from two popular APIs (TMDb and OMDb). Users can input a movie title and get a predicted rating category along with information such as release date, runtime, plot, actors, and IMDb ratings.

## Features
- **Movie Rating Prediction**: Based on a dataset of movie ratings, it predicts whether a movie is "Good," "Neutral," or "Bad."
- **Movie Details**: Fetches detailed movie information from:
  - **TMDb API**: Provides title, release date, overview, runtime, and platforms where the movie is available.
  - **OMDb API**: Fetches title, year, plot, actors, IMDb rating, and runtime.
- **Visual Design**: Includes a cool background, stylized headers, and dynamic movie information display.

## Installation

### Prerequisites
To run the app, make sure you have Python 3.8 or higher installed. You will also need the following libraries:

- `streamlit`
- `pandas`
- `pickle`
- `requests`
- `scikit-learn`
- `tabulate`

### Steps to Install

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/movie-rating-prediction.git
   cd movie-rating-prediction
   ```

## APIs Used

- **TMDb API**: [TMDb Documentation](https://www.themoviedb.org/documentation/api)
- **OMDb API**: [OMDb Documentation](http://www.omdbapi.com/)

Make sure to get an API key for both TMDb and OMDb and replace the API keys in the `app.py` file.

## Usage

1. **Enter a Movie Title**: In the provided input field, type the name of the movie you want to search for.
2. **View Predicted Rating**: The app will display whether the movie is categorized as "Good," "Neutral," or "Bad" based on the dataset.
3. **View Detailed Information**:
   - **TMDb Details**: Movie title, release date, overview, runtime, and available platforms.
   - **OMDb Details**: Movie title, year, plot, actors, IMDb rating, and runtime.

### Example

- **Input**: `Inception`
- **Output**:
  - **Predicted Category**: Good
  - **TMDb Details**: Title, Overview, Platforms, etc.
  - **OMDb Details**: Plot, Actors, IMDb Rating, etc.
