import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

############################################
# 1. Auth Handler (Log in & Sign up)
############################################

def init_auth_state():
    """Ensure auth‑related keys exist in session_state."""
    if "users" not in st.session_state:
        # pre‑seed with an admin account
        st.session_state["users"] = {"admin": "1234"}
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False


def login_screen():
    """Render Log in / Sign up UI.

    * Radio selector lets user toggle between the two modes.
    * After a successful log in or sign up we mark `logged_in` and rerun so
      the main app can render immediately without an extra click."""

    st.title("Welcome")
    mode = st.radio("Choose an option", ["Log in", "Sign up"], horizontal=True)

    if mode == "Log in":
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            users = st.session_state["users"]
            if username in users and users[username] == password:
                st.session_state["logged_in"] = True
                try:
                    st.experimental_rerun()
                except RuntimeError:
                    pass
            else:
                st.error("Invalid credentials. Please try again.")

    else:  # Sign up
        with st.form("signup_form", clear_on_submit=False):
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password", type="password")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if not new_user or not new_pass:
                st.error("Username and password cannot be empty.")
            elif new_user in st.session_state["users"]:
                st.error("Username already exists. Please pick another.")
            else:
                st.session_state["users"][new_user] = new_pass
                st.success("Account created! You are now logged in.")
                st.session_state["logged_in"] = True
                try:
                    st.experimental_rerun()
                except RuntimeError:
                    pass

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

    # Include all actor names to improve similarity on casts
    star_tokens = "".join(" " + row[col] for col in ["Star1", "Star2", "Star3", "Star4"])

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
        "Star2",
        "Star3",
        "Star4",
        "Poster_Link",
        "Meta_score",
        "Overview",
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
# 4. Recommendation Helper (UI)
############################################

def display_recommendations(title: str, df: pd.DataFrame, cosine_sim, indices):
    recs_df = get_recommendations(title, df, cosine_sim, indices)
    if recs_df is None:
        st.warning(f"'{title}' not found in dataset.")
    else:
        st.success(f"Movies similar to '{title}':")
        for _, row in recs_df.iterrows():
            col_img, col_text = st.columns([1, 5])
            with col_img:
                if row["Poster_Link"]:
                    st.image(row["Poster_Link"], width=120)
            with col_text:
                actors = ", ".join(filter(None, (row.get(col, "") for col in ["Star1", "Star2", "Star3", "Star4"]))) or "N/A"

                overview = row.get("Overview", "")
                if len(overview) > 500:
                    overview = overview[:480].rstrip() + "…"

                st.markdown(
                    f"**{row['Series_Title']}**\n\n"
                    f"Released: {row['Released_Year']}  |  "
                    f"IMDb: {row['IMDB_Rating']}⭐\n\n"
                    f"Director: {row['Director']}\n\n"
                    f"Actors: {actors}\n\n"
                    f"Genre: {row['Genre']}\n\n"
                    f"**Overview:** {overview}"
                )

############################################
# 5. Main Recommender UI
############################################

def main_app(df, cosine_sim, indices):
    st.title("🎬 Movie Recommender")
    st.write(
        "Enter a movie you love and press **Enter** to get five similar gems – complete with posters and key details."
    )

    def recommend_callback():
        title = st.session_state.get("user_movie", "").strip()
        if title:
            display_recommendations(title, df, cosine_sim, indices)

    st.text_input("Movie Title:", key="user_movie", on_change=recommend_callback)

############################################
# 6. Recommendation Logic
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
                "Star2",
                "Star3",
                "Star4",
                "IMDB_Rating",
                "Genre",
                "Overview",
                "Poster_Link",
            ]
        ]
        .reset_index(drop=True)
    )

############################################
# 7. App bootstrap
############################################

def run():
    init_auth_state()

    if not st.session_state["logged_in"]:
        login_screen()
        if not st.session_state["logged_in"]:
            st.stop()

    df, cosine_sim, indices = load_data()
    main_app(df, cosine_sim, indices)

if __name__ == "__main__":
    run()
