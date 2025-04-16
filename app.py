import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

############################################
#      SIMPLE LOGIN HANDLER (EXAMPLE)      #
############################################

def login_screen():
    """
    Shows a login screen where the user inputs username and password.
    If correct, st.session_state["logged_in"] is set to True.
    """
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        # Replace these with a real authentication check, e.g. comparing a hashed password from a DB
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")

############################################
#  2. Dataset Loading & Weighted Features  #
############################################

def create_weighted_features(row):
    """
    Convert numeric fields (IMDB_Rating, Meta_score) into repeated tokens,
    and replicate text fields (Genre, Director) a chosen number of times
    to achieve weighting in TF-IDF.
    """
    # Convert IMDB Rating (0–10) to an int
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

def load_data():
    """
    Loads the CSV dataset and returns (df, cosine_sim, indices)
    after building the weighted features and computing similarity.
    """
    # Adjust to your local path and filename
    DATA_PATH = "/path/to/your/folder"
    CSV_FILE = "imdb_top_1000.csv"
    csv_path = os.path.join(DATA_PATH, CSV_FILE)

    df = pd.read_csv(csv_path)

    # Fill missing values
    for col in ["Director", "Genre", "IMDB_Rating", "Meta_score"]:
        df[col] = df[col].fillna("")

    df["weighted_features"] = df.apply(create_weighted_features, axis=1)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["weighted_features"])

    # Cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Map titles to DataFrame indices
    df["Series_Title_lower"] = df["Series_Title"].str.lower()
    indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()

    return df, cosine_sim, indices

############################################
#  3. Recommender Function
############################################

def get_recommendations(title, df, cosine_sim, indices):
    """
    Returns a Series of the top 10 recommended movie titles for the given title.
    """
    title_lower = title.lower()
    if title_lower not in indices:
        return None

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity descending
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the current movie
    sim_scores = sim_scores[1:]
    # Top 10
    movie_indices = [i[0] for i in sim_scores[:10]]
    return df.iloc[movie_indices]["Series_Title"]

############################################
#  4. Main App
############################################

def main_app(df, cosine_sim, indices):
    st.title("Movie Recommender")
    st.write("Enter a movie title and get similar recommendations.")

    user_movie = st.text_input("Movie Title:", value="The Godfather")

    if st.button("Recommend"):
        recommendations = get_recommendations(user_movie, df, cosine_sim, indices)
        if recommendations is None:
            st.warning(f"'{user_movie}' not found in dataset.")
        else:
            st.success(f"Movies similar to '{user_movie}':")
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")

def main():
    # Check if we've already logged in
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        # Display login screen
        login_screen()
    else:
        # If logged in, load data and show the recommender
        df, cosine_sim, indices = load_data()
        main_app(df, cosine_sim, indices)

if __name__ == "__main__":
    main()
