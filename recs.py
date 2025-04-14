import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import kagglehub


path = "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
csv_path = os.path.join(path, "imdb_top_1000.csv")

df = pd.read_csv(csv_path)

# Fill NaN with empty string to avoid errors
for col in ["Director", "Genre", "IMDB_Rating", "Meta_score", "Star1"]:
  df[col] = df[col].fillna("")

# Create a combined string:
df["combined_features"] = (
    df["Director"] + " " +
    df["Genre"] + " " +
    str(df["IMDB_Rating"]) + " " +
    str(df["Meta_score"]) + " " +
    df["Star1"]
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# For quick lookups, create a Series mapping titles to their DataFrame index
# Lowercase the title to make matching easier
df["Series_Title_lower"] = df["Series_Title"].str.lower()
indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # Convert to lowercase to match the index
    title_lower = title.lower()
    
    if title_lower not in indices:
        print(f"'{title}' not found in dataset.")
        return []
    
    # Get the index of the movie that matches the title
    idx = indices[title_lower]

    # Retrieve pairwise similarity scores for this movie to all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # The first element is the movie itself; we exclude it
    sim_scores = sim_scores[1:]
    
    # Get the indices of the top 10 most similar movies
    top_movie_indices = [i[0] for i in sim_scores[:10]]
    
    # Return the top 10 most similar movies
    return df.iloc[top_movie_indices]["Series_Title"]


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a movie title (or 'quit' to stop): ")
        if user_input.lower() == "quit":
            break

        recommendations = get_recommendations(user_input, cosine_sim, df, indices)
        if recommendations.empty:
          print(f"'{user_input}' not found or no similar movies.")
        else:
          print(f"\nMovies similar to '{user_input}':")
          for i, rec_title in enumerate(recommendations, start=1):
            print(f"{i}. {rec_title}")

