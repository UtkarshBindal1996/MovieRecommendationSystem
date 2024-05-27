import streamlit as st
import pandas as pd

# Sample movie data (replace with your actual data source)
movies = pd.DataFrame({
    "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
    "genre": ["Action", "Comedy", "Romance", "Sci-Fi", "Thriller"]
})

# User input dictionary (initially empty)
user_ratings = {}

def recommend_movies(user_ratings, movies):
  """
  Recommends movies based on cosine similarity between user ratings and movie data.

  Args:
      user_ratings: A dictionary storing user ratings for movie IDs (key) and ratings (value).
      movies: A pandas DataFrame containing movie information (title, genre, etc.).

  Returns:
      A list of top N recommended movie titles.
  """

  # Create a DataFrame from user ratings (assuming movie IDs are unique)
  user_ratings_df = pd.DataFrame.from_dict(user_ratings, orient='index', columns=['rating'])

  # Calculate cosine similarity between user ratings and movie data (assuming movie IDs are the same)
  similarity_matrix = cosine_similarity(user_ratings_df, movies)

  # Get average rating for each movie (optional, for filtering)
  average_ratings = movies.groupby('title')['rating'].transform('mean')

  # Select top N movies with high similarity and (optional) above average rating
  N = 5  # Number of recommendations
  recommendations = similarity_matrix.iloc[0].sort_values(ascending=False).head(N).index[similarity_matrix.iloc[0] > 0.5]  # Filter by minimum similarity threshold
  recommendations = recommendations[recommendations.isin(average_ratings[average_ratings > user_ratings_df.iloc[0]['rating']].index)]  # Optional filtering by average rating

  return sorted(list(recommendations))  # Sort recommendations alphabetically

st.title("Movie Recommendation App")

# User input section
all_ratings = []  # List to store all movie ratings

movies_to_rate = st.multiselect("Select movies you want to rate:", movies["title"].to_list())

for movie in movies_to_rate:
  # Check if user selected a movie
  if movie:
    movie_column, rating_column = st.columns(2)
    with movie_column:
      st.write(f"Movie: {movie}")  # Display selected movie for clarity
    with rating_column:
      rating_select = rating_column.slider(f"Rate {movie}:", 1, 5, 1)
    all_ratings.append((movie, rating_select))  # Add rating to list

if st.button("Submit Ratings"):
  # Update user_ratings dictionary from all_ratings list
  for movie, rating in all_ratings:
    user_ratings[movie] = rating

# Recommendation section
if user_ratings:
  recommendations = recommend_movies(user_ratings, movies.copy())
  st.subheader("Recommended Movies for You:")
  for movie in recommendations:
    st.write(movie)
else:
  st.write("Select some movies and rate them to get recommendations!")
