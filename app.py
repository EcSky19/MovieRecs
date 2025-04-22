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
            st.stop()  # rerun with new state on next interaction
        else:
            st.error("Incorrect username or password")

############################################
# 2. Weighted Feature Creation
############################################

def create_weighted_features(row):
    """Create synthetic text features with hand‑tuned weights so that
    the vectoriser can pick them up when computing similarity."""
    try:
        rating_int = int(round(float(row["IMDB_Rating"])))
    except ValueError:
        rating_int = 0

    try:
        meta_int = int(round(float(row.get("Meta_score", 0)) / 10))
    except ValueError:
        meta_int = 0

    rating_tokens = " rating" * rating_int
    metascore_tokens = " metascore" * meta_int
    genre_tokens = (" " + row["Genre"]) * 3
    director_tokens = (" " + row["Director"]) * 1
    star_tokens = (" " + row["Star1"]) * 1

    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens + star_tokens).strip()

############################################
# 3. Load Data and Compute Similarity
############################################

def load_data():
    DATA_PATH = (
        "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/"
        "imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
    )
    CSV_FILE = "imdb_top_1000.csv"
    csv_path = os.path.join(DATA_PATH, CSV_FILE)

    df = pd.read_csv(csv_path)

    # Ensure required columns exist and have no NAs
    needed_cols = [
        "Series_Title",
        "Released_Year",
        "Genre",
        "IMDB_Rating",
        "Director",
        "Star1",
        "Poster_Link",
        "Meta_score",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # Feature engineering for similarity search
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

def get_recommendations(title: str, df: pd.DataFrame, cosine_sim, indices):
    title_lower = title.lower().strip()
    if title_lower not in indices:
        return None

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:6]]  # top‑5 similar

    return (
        df.iloc[top_indices][
            [
                "Series_Title",
                "Released_Year",
                "Director",
                "Star1",
                "IMDB_Rating",
                "Genre",
                "Poster_Link",
            ]
        ]
        .reset_index(drop=True)
        .rename(columns={"Star1": "Lead_Actor"})
    )

############################################
# 5. Main Recommender UI
############################################

def main_app(df, cosine_sim, indices):
    st.title("🎬 Movie Recommender")
    st.write("Enter a movie you love and discover five similar gems – complete with posters and key details.")

    user_movie = st.text_input("Movie Title:")

    if st.button("Recommend"):
        recs_df = get_recommendations(user_movie, df, cosine_sim, indices)
        if recs_df is None:
            st.warning(f"'{user_movie}' not found in dataset.")
        else:
            st.success(f"Movies similar to '{user_movie}':")
            for _, row in recs_df.iterrows():
                col_img, col_text = st.columns([1, 5])
                with col_img:
                    if row["Poster_Link"]:
                        st.image(row["Poster_Link"], width=120)
                with col_text:
                    st.markdown(
                        f"**{row['Series_Title']}**\n\n"
                        f"Released: {row['Released_Year']}  |  "
                        f"IMDb: {row['IMDB_Rating']}⭐\n\n"
                        f"Director: {row['Director']}  |  "
                        f"Lead Actor: {row['Lead_Actor']}\n\n"
                        f"Genre: {row['Genre']}"
                    )

############################################
# 6. App bootstrap
############################################

def run():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_screen()
        st.stop()
    else:
        df, cosine_sim, indices = load_data()
        main_app(df, cosine_sim, indices)

if __name__ == "__main__":
    run()
