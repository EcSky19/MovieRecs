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
            # End this script execution now so that the next rerun picks up the new state
            st.stop()
        else:
            st.error("Incorrect username or password")

############################################
# 2. Weighted Feature Creation
############################################

def create_weighted_features(row):
    """Create synthetic text features with hand‑tuned weights for rating, metascore,
    genre, director and main star so that the vectoriser can pick them up."""
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
    star_tokens = (" " + row["Star1"]) * 1

    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens + star_tokens).strip()

############################################
# 3. Load Data and Compute Similarity
############################################

def load_data():
    DATA_PATH = "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
    CSV_FILE = "imdb_top_1000.csv"
    csv_path = os.path.join(DATA_PATH, CSV_FILE)

    df = pd.read_csv(csv_path)

    # Clean up NA fields we rely on later
    for col in ["Director", "Genre", "IMDB_Rating", "Meta_score", "Star1", "Poster_Link"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            # Back‑stop: if the dataset is missing Poster_Link create an empty column so the rest of the app works
            if col == "Poster_Link":
                df[col] = ""

    # Pre‑compute vector representations for similarity search
    df["weighted_features"] = df.apply(create_weighted_features, axis=1)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["weighted_features"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Speedy lookup from lower‑cased title → index
    df["Series_Title_lower"] = df["Series_Title"].str.lower()
    indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()

    return df, cosine_sim, indices

############################################
# 4. Recommendation Function
############################################

def get_recommendations(title: str, df: pd.DataFrame, cosine_sim, indices):
    """Return a dataframe with the 5 most similar movies, including their poster link."""
    title_lower = title.lower().strip()
    if title_lower not in indices:
        return None

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:6]]  # Skip first (the movie itself)

    return df.iloc[top_indices][["Series_Title", "Poster_Link"]].reset_index(drop=True)

############################################
# 5. Main App
############################################

def main_app(df, cosine_sim, indices):
    st.title("🎬 Movie Recommender")
    st.write("Enter a movie title and we'll surface 5 similar films — posters included!")

    user_movie = st.text_input("Movie Title:", value="")

    if st.button("Recommend"):
        recs_df = get_recommendations(user_movie, df, cosine_sim, indices)
        if recs_df is None:
            st.warning(f"'{user_movie}' not found in dataset.")
        else:
            st.success(f"Movies similar to '{user_movie}':")
            for _, row in recs_df.iterrows():
                col_img, col_text = st.columns([1, 4])
                with col_img:
                    if row["Poster_Link"]:
                        st.image(row["Poster_Link"], width=120)
                with col_text:
                    st.markdown(f"**{row['Series_Title']}**")

############################################
# 6. App bootstrap
############################################

def run():
    # Initialise session key on first run
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_screen()
        # Stop the script so that Streamlit re‑runs after any user interaction
        st.stop()
    else:
        df, cosine_sim, indices = load_data()
        main_app(df, cosine_sim, indices)

if __name__ == "__main__":
    run()
