import os
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

########################
# Load the dataset     #
########################

# Adjust the path/filename as needed
DATA_PATH = "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
CSV_FILE = "imdb_top_1000.csv"
csv_path = os.path.join(DATA_PATH, CSV_FILE)

df = pd.read_csv(csv_path)

# Fill missing values
for col in ["Director", "Genre", "IMDB_Rating", "Meta_score"]:
    df[col] = df[col].fillna("")

#########################################
# Weighted feature creation function    #
#########################################
def create_weighted_features(row):
    """
    Convert numeric fields (IMDB_Rating, Meta_score) into repeated tokens,
    and replicate text fields (Genre, Director) a chosen number of times
    to achieve weighting in TF-IDF.
    """
    # Convert IMDB_Rating (0–10) to an int
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
    rating_tokens = " rating" * rating_int
    metascore_tokens = " metascore" * meta_int
    # Suppose we give Genre weight=3, Director weight=1
    genre_tokens = (" " + row["Genre"]) * 3
    director_tokens = (" " + row["Director"]) * 1
    
    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens).strip()

#########################################
# Apply weighting and build vectors     #
#########################################

df["weighted_features"] = df.apply(create_weighted_features, axis=1)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["weighted_features"])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Map titles to DataFrame indices for quick lookup
df["Series_Title_lower"] = df["Series_Title"].str.lower()
indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()

#########################################
# Recommendation helper function        #
#########################################
def get_recommendations(title, df, indices, cosine_sim):
    # Lowercase the input to match 'Series_Title_lower'
    title_lower = title.lower()
    
    if title_lower not in indices:
        return None  # indicates not found in dataset
    
    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort descending by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the movie itself
    sim_scores = sim_scores[1:]
    
    # Top 10
    movie_indices = [i[0] for i in sim_scores[:10]]
    return df.iloc[movie_indices]["Series_Title"]

#########################################
# Streamlit UI                          #
#########################################
def main():
    st.title("Movie Recommender")
    st.write("Enter a movie title and get similar recommendations.")

    user_movie = st.text_input("Enter a movie title", value="")

    if st.button("Recommend"):
        recommendations = get_recommendations(user_movie, df, indices, cosine_sim)
        if recommendations is None:
            st.warning("Movie not found in dataset. Please try a different title.")
        else:
            st.success(f"Movies similar to '{user_movie}':")
            for i, title in enumerate(recommendations, start=1):
                st.write(f"{i}. {title}")

if __name__ == "__main__":
    main()
