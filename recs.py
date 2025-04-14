import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the data
path = "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
csv_path = os.path.join(path, "imdb_top_1000.csv")

df = pd.read_csv(csv_path)

# Ensure the columns needed are present; fill missing values
for col in ["Director", "Genre", "IMDB_Rating", "Meta_score"]:
    df[col] = df[col].fillna("")

def create_weighted_features(row):
    """
    Convert numeric fields (IMDB_Rating, Meta_score) into repeated tokens,
    and replicate text fields (Genre, Director) a chosen number of times
    to achieve weighting in TF-IDF.
    """
    # Convert rating (0–10) to an int
    try:
        rating_int = int(round(float(row["IMDB_Rating"])))
    except ValueError:
        rating_int = 0
    
    # Convert Meta_score (0–100) by dividing by 10
    try:
        meta_int = int(round(float(row["Meta_score"]) / 10))
    except ValueError:
        meta_int = 0
    
    # Weighted repetition
    rating_tokens = (" rating" * rating_int)
    metascore_tokens = (" metascore" * meta_int)
    # Suppose we give Genre weight=3, Director weight=1
    genre_tokens = (" " + row["Genre"]) * 3
    director_tokens = (" " + row["Director"]) * 1
    
    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens).strip()

# Apply the weighting logic to every row
df["weighted_features"] = df.apply(create_weighted_features, axis=1)

# Build TF-IDF matrix from these weighted features
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["weighted_features"])

# Compute the cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

df["Series_Title_lower"] = df["Series_Title"].str.lower()
indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices):
    title_lower = title.lower()
    if title_lower not in indices:
        print(f"'{title}' not found in dataset.")
        return pd.Series([], dtype=object)
    
    # Get the index of the movie that matches the title
    idx = indices[title_lower]

    # Retrieve similarity scores for this movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # The first element is the movie itself; exclude it
    sim_scores = sim_scores[1:]
    
    # Get the indices of the top 10 most similar
    top_movie_indices = [i[0] for i in sim_scores[:10]]
    return df.iloc[top_movie_indices]["Series_Title"]

# Interactive prompt
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a movie title (or 'quit' to stop): ")
        if user_input.lower() == "quit":
            break

        recommendations = get_recommendations(user_input, cosine_sim, df, indices)
        if len(recommendations) == 0:
            print(f"'{user_input}' not found or no similar movies.")
        else:
            print(f"\nMovies similar to '{user_input}':")
            for i, rec_title in enumerate(recommendations, start=1):
                print(f"{i}. {rec_title}")
