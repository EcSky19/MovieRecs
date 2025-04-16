# app.py

import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

############################################
# 1. Load the dataset
############################################

DATA_PATH = "/path/to/dataset/folder"  # Adjust to your folder
CSV_FILE = "imdb_top_1000.csv"         # Adjust if needed

csv_path = os.path.join(DATA_PATH, CSV_FILE)
df = pd.read_csv(csv_path)

# Fill missing values
for col in ["Director", "Genre", "IMDB_Rating", "Meta_score"]:
    df[col] = df[col].fillna("")

############################################
# 2. Create weighted features
############################################

def create_weighted_features(row):
    try:
        rating_int = int(round(float(row["IMDB_Rating"])))
    except ValueError:
        rating_int = 0

    try:
        meta_int = int(round(float(row["Meta_score"]) / 10))
    except ValueError:
        meta_int = 0

    rating_tokens = " rating" * rating_int
    metascore_tokens = " metascore" * meta_int
    genre_tokens = (" " + row["Genre"]) * 3
    director_tokens = (" " + row["Director"]) * 1

    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens).strip()

df["weighted_features"] = df.apply(create_weighted_features, axis=1)

############################################
# 3. Vectorize and compute similarity
############################################

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["weighted_features"])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

df["Series_Title_lower"] = df["Series_Title"].str.lower()
indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()

############################################
# 4. Recommendation function
############################################

def get_recommendations(title):
    title_lower = title.lower()
    if title_lower not in indices:
        return None  # indicates not found
    
    idx = indices[title_lower]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # exclude itself

    top_indices = [i[0] for i in sim_scores[:10]]
    return df.iloc[top_indices]["Series_Title"]

############################################
# 5. Streamlit UI
############################################

def main():
    st.title("Movie Recommender")
    st.write("Enter a movie title and get similar recommendations.")

    user_movie = st.text_input("Movie Title:", value="")

    if st.button("Recommend"):
        recommendations = get_recommendations(user_movie)
        if recommendations is None:
            st.warning(f"'{user_movie}' not found in dataset.")
        else:
            st.success(f"Movies similar to '{user_movie}':")
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")

if __name__ == "__main__":
    main()
