# ============================================
# Movie Recommendation System (CLI Based)
# Made By: Vedant
# ============================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Movie Dataset

movies = {
    "Movie": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Avengers Endgame",
        "Titanic",
        "Fast and Furious",
        "The Notebook",
        "Doctor Strange",
        "Iron Man",
        "Joker"
    ],

    "Genre": [
        "Sci-Fi Thriller Mind-Bending",
        "Sci-Fi Space Emotional",
        "Action Crime Hero",
        "Action Superhero Sci-Fi",
        "Romance Drama Emotional",
        "Action Assassin Crime",
        "Romance Love Drama",
        "Magic Action Sci-Fi",
        "Technology Superhero Action",
        "Psychological Crime Drama"
    ]
}

df = pd.DataFrame(movies)

# Convert Text Data into Numerical Form

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df["Genre"])

# Similarity Calculation

similarity = cosine_similarity(tfidf_matrix)

# Recommendation Function

def recommend(movie_name):

    movie_name = movie_name.lower()

    if movie_name not in df["Movie"].str.lower().values:
        print("\nMovie not found in database.")
        return

    index = df[df["Movie"].str.lower() == movie_name].index[0]

    similarity_scores = list(enumerate(similarity[index]))

    sorted_movies = sorted(similarity_scores,
                           key=lambda x: x[1],
                           reverse=True)

    print("\nRecommended Movies:\n")

    count = 0

    for movie in sorted_movies[1:]:

        movie_index = movie[0]

        print(df.iloc[movie_index]["Movie"])

        count += 1

        if count == 5:
            break


# Main Program

print("===================================")
print("     MOVIE RECOMMENDATION SYSTEM")
print("===================================")

print("\nAvailable Movies:\n")

for movie in df["Movie"]:
    print("-", movie)

user_movie = input("\nEnter movie name: ")

recommend(user_movie)