import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

############################################
# 1. Simple Login Handler (No experimental_rerun)
############################################

def login_screen():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            # We don't call st.experimental_rerun(). Instead:
            # Just stop the script here so that on the next
            # user interaction, the app re-runs with new state.
            st.stop()  # End this script execution now
        else:
            st.error("Incorrect username or password")

############################################
# 2. Weighted Feature Creation
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

    rating_tokens = (" rating" * rating_int)
    metascore_tokens = (" metascore" * meta_int)
    genre_tokens = (" " + row["Genre"]) * 3
    director_tokens = (" " + row["Director"]) * 1

    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens).strip()

############################################
# 3. Load Data and Compute Similarity
############################################

def load_data():
    DATA_PATH = "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
    CSV_FILE = "imdb_top_1000.csv"
    csv_path = os.path.join(DATA_PATH, CSV_FILE)

    df = pd.read_csv(csv_path)

    for col in ["Director", "Genre", "IMDB_Rating", "Meta_score"]:
        df[col] = df[col].fillna("")

    df["weighted_features"] = df.apply(create_weighted_features, axis=1)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["weighted_features"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    df["Series_Title_lower"] = df["Series_Title"].str.lower()
    indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()

    return df, cosine_sim, indices

############################################
# 4. Recommendation Function
############################################

def get_recommendations(title, df, cosine_sim, indices):
    title_lower = title.lower()
    if title_lower not in indices:
        return None

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    top_indices = [i[0] for i in sim_scores[:5]]
    return df.iloc[top_indices]["Series_Title"]

############################################
# 5. Main App
############################################

def main_app(df, cosine_sim, indices):
    st.title("Movie Recommender")
    st.write("Enter a movie title and get similar recommendations.")

    user_movie = st.text_input("Movie Title:", value="")

    if st.button("Recommend"):
        recommendations = get_recommendations(user_movie, df, cosine_sim, indices)
        if recommendations is None:
            st.warning(f"'{user_movie}' not found in dataset.")
        else:
            st.success(f"Movies similar to '{user_movie}':")
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")

def main():
    # Initialize "logged_in" in session_state if it doesn't exist
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        # Show the login screen
        login_screen()
        # If the user does NOT click the button, script continues
        # So we call st.stop() to not load data below yet
        st.stop()
    else:
        # Already logged in, so load data & show the recommender
        df, cosine_sim, indices = load_data()
        main_app(df, cosine_sim, indices)

if __name__ == "__main__":
    main()
