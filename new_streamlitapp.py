import streamlit as st
import pandas as pd
import pickle
import numpy as np
import datetime

# Fetching the stored data.
rating_data_path = "./data/rating.csv"
rating_df = pd.read_csv(rating_data_path)

movie_data_path = "./data/movie.csv"
movie_df = pd.read_csv(movie_data_path)

# Filtering out the data.
rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'])
rating_df = rating_df[rating_df['timestamp'].dt.year >= 2010]


# Filtering the relevant movies
unique_movies_id = rating_df['movieId'].unique().tolist()
unique_movies_names = []
movie_df = movie_df.set_index('movieId')
for movie in unique_movies_id:
    unique_movies_names.append(movie_df.iloc[movie]["title"])

# Fetching the simiilarity matrix
with open('./data/similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

with open('./data/clustered_data.pkl', 'rb') as f:
    clustered_df = pickle.load(f)

# Adjusted cosine similarity
# function to compute similarity between 2 users
def adjusted_cosine_similarity(user1_ratings, user2_ratings, min_ratings=7):
    """
    Calculates the adjusted cosine similarity between two user rating dictionaries,
    considering only movies rated by both users and enforcing a minimum rating threshold.
    
    Args:
      user1_ratings: A dictionary representing user 1's ratings (movie ID as key, rating as value).
      user2_ratings: A dictionary representing user 2's ratings (movie ID as key, rating as value).
      min_ratings: Minimum number of movies required to be rated by both users for similarity calculation (default 1).
    
    Returns:
      The adjusted cosine similarity between the two users, or 0 if the minimum rating threshold is not met or no movies are rated by both users.
    """
    
    intersection = set(user1_ratings.keys()) & set(user2_ratings.keys())  # Get common movies
    if len(intersection) < min_ratings:
        return 0  # Similarity is 0 if below minimum rating threshold
    
    if not intersection:
        return 0  # No common movies, similarity is 0
    
    user1_avg = np.mean(list(user1_ratings[movie][0] for movie in intersection))  # Average rating for user 1 (using list comprehension)
    user2_avg = np.mean(list(user2_ratings[movie][0] for movie in intersection))  # Average rating for user 2 (using list comprehension)

    # Define a function to calculate freshness penalty (replace with your chosen approach)
    def freshness_penalty(time_difference):
        decay_rate = 0.01  # Adjust this parameter to control the penalty strength
        time_difference_days = time_difference.days
        penalty = 1 - decay_rate * time_difference_days
        return max(penalty, 0)  # Ensure penalty is between 0 and 1
    
    # numerator = sum((user1_ratings[movie] - user1_avg) * (user2_ratings[movie] - user2_avg) for movie in intersection)
    # denominator = np.sqrt(sum((user1_ratings[movie] - user1_avg)**2 for movie in intersection) * 
    #                      sum((user2_ratings[movie] - user2_avg)**2 for movie in intersection))

    numerator = sum([(user1_ratings[movie][0] - user1_avg) * (user2_ratings[movie][0] - user2_avg) *
                   freshness_penalty(user1_ratings[movie][1] - user2_ratings[movie][1]) for movie in intersection])
    denominator = np.sqrt(sum((user1_ratings[movie][0] - user1_avg)**2 for movie in intersection) *
                          sum((user2_ratings[movie][0] - user2_avg)**2 for movie in intersection))
    
    if denominator == 0:
        return 0  # Avoid division by zero
    
    return numerator / denominator

# User input dictionary (initially empty)
user_ratings = {}

def recommend_movies(user_ratings, movies):
    # Find most similar user in training data (assuming similarity scores are stored somewhere)
    most_similar_user_index = np.argmax(similarity_matrix, axis=0)  # Find index of max similarity
    # Predict cluster label based on most similar user's cluster
    predicted_cluster = clustered_df.loc[most_similar_user_index, 'cluster_id'].iloc[0]
    recommendations = {}

    # Need to recommend high rated movies from the same cluster.
    # Filter out all entries from a particular cluster
    single_cluster_df = clustered_df[clustered_df['cluster_id'] == predicted_cluster]

    # Create a priority list of the (on average) highest rated movies in that cluster
    # Need to create a dictionary from the rating values in that cluster.
    result_dict = {}
    for _, row in single_cluster_df.iterrows():
        similarity_score = adjusted_cosine_similarity(row['movie_rating_dict'], user_ratings, min_ratings=1)
        for movieId, rating in row['movie_rating_dict'].items():
            if movieId not in result_dict:
                result_dict[movieId] = {
                    "sum_similarity_rating": 0,
                    "sum_similarity": 0
                }
            result_dict[movieId]["sum_similarity_rating"] += similarity_score*rating
            result_dict[movieId]["sum_similarity"] += similarity_score

            if result_dict[movieId]["sum_similarity"] == 0:
                net_score = 0
            else:
                net_score = result_dict[movieId]["sum_similarity_rating"]/result_dict[movieId]["sum_similarity"]
            recommendations[movieId] = net_score

st.title("Movie Mate: Find your next cinematic match")
st.write("Simply rate a couple of movies from the list below and hit get recomendation!")

# User input section
all_ratings = []  # List to store all movie ratings

movies_to_rate = st.multiselect("Select movies you want to rate:", unique_movies_names)
submit_button_disabled = not movies_to_rate  # Initially disabled if no movies selected

for movie in movies_to_rate:
  # Check if user selected a movie
  if movie:
    movie_column, rating_column = st.columns(2)
    with movie_column:
      st.write(f"Movie: {movie}")  # Display selected movie for clarity
    with rating_column:
      rating_select = rating_column.slider(f"Rate {movie}:", 1, 5, 1)
    all_ratings.append((movie, rating_select))  # Add rating to list
    submit_button_disabled = False

if st.button("Submit Ratings", disabled=submit_button_disabled):
    # Update user_ratings dictionary from all_ratings list
    for movie, rating in all_ratings:
        movie_id = movie_df[movie_df['title'] == movie].index[0]
        user_ratings[movie_id] = [rating, datetime.datetime.now()]

# Recommendation section
recommendations = recommend_movies(user_ratings, movies.copy())
st.subheader("Recommended Movies for You:")
for movie in recommendations:
    st.write(movie)
