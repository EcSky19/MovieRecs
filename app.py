import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

############################################
# 0. Utility
############################################

def safe_rerun() -> None:
    """Attempt `st.experimental_rerun()` only when running inside the
    Streamlit runtime. Silently ignore the call in plain‑Python mode."""
    try:
        st.experimental_rerun()
    except Exception:
        pass

############################################
# 1. Auth Handler (Log in & Sign up — buttons only)
############################################

def init_auth_state():
    """Seed a default admin user and ensure auth keys exist."""
    st.session_state.setdefault("users", {"admin": "1234"})
    st.session_state.setdefault("logged_in", False)


def login_screen():
    """Render Log in / Sign up UI.

    * Uses **buttons** so the **Enter key** has no effect.
    * `st.radio` is given an explicit `key` to avoid duplicate‑ID errors on
      reruns.
    * After success we set `logged_in` and invoke `safe_rerun()` so the main
      app renders immediately under Streamlit.
    """

    st.title("Welcome")
    mode = st.radio("Choose an option", ["Log in", "Sign up"], horizontal=True, key="auth_mode")

    if mode == "Log in":
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Log in", key="login_btn"):
            users = st.session_state["users"]
            if username in users and users[username] == password:
                st.session_state["logged_in"] = True
                safe_rerun()
            else:
                st.error("Invalid credentials. Please try again.")

    else:  # Sign up
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")

        if st.button("Create account", key="signup_btn"):
            if not new_user or not new_pass:
                st.error("Username and password cannot be empty.")
            elif new_user in st.session_state["users"]:
                st.error("Username already exists. Please pick another.")
            else:
                st.session_state["users"][new_user] = new_pass
                st.success("Account created! You are now logged in.")
                st.session_state["logged_in"] = True
                safe_rerun()

############################################
# 2. Weighted Feature Creation
############################################

def create_weighted_features(row):
    """Create synthetic text for TF‑IDF based on weighted attributes."""
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
    director_tokens = (" " + row["Director"])
    star_tokens = "".join(" " + row[col] for col in ["Star1", "Star2", "Star3", "Star4"])

    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens + star_tokens).strip()

############################################
# 3. Load Data & Similarity Matrix
############################################

def load_data():
    DATA_PATH = (
        "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/"
        "imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
    )
    CSV_FILE = "imdb_top_1000.csv"
    csv_path = os.path.join(DATA_PATH, CSV_FILE)

    df = pd.read_csv(csv_path)

    required = [
        "Series_Title","Released_Year","Genre","IMDB_Rating","Director",
        "Star1","Star2","Star3","Star4","Poster_Link","Meta_score","Overview"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["weighted_features"] = df.apply(create_weighted_features, axis=1)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["weighted_features"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    df["Series_Title_lower"] = df["Series_Title"].str.lower()
    indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()
    return df, cosine_sim, indices

############################################
# 4. Recommendation Helpers
############################################

def get_recommendations(title: str, df: pd.DataFrame, cosine_sim, indices):
    key = title.lower().strip()
    if key not in indices:
        return None
    idx = indices[key]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    top_idx = [i[0] for i in sim_scores[1:6]]
    cols = [
        "Series_Title","Released_Year","Director","Star1","Star2","Star3","Star4",
        "IMDB_Rating","Genre","Overview","Poster_Link"
    ]
    return df.iloc[top_idx][cols].reset_index(drop=True)


def display_recommendations(title, df, cosine_sim, indices):
    recs = get_recommendations(title, df, cosine_sim, indices)
    if recs is None:
        st.warning(f"'{title}' not found in dataset.")
        return
    st.success(f"Movies similar to '{title}':")
    for _, row in recs.iterrows():
        col_img, col_txt = st.columns([1,5])
        with col_img:
            if row["Poster_Link"]:
                st.image(row["Poster_Link"], width=120)
        with col_txt:
            actors = ", ".join(filter(None, (row[col] for col in ["Star1","Star2","Star3","Star4"]))) or "N/A"
            overview = row["Overview"][:480].rstrip() + ("…" if len(row["Overview"])>480 else "")
            st.markdown(
                f"**{row['Series_Title']}**\n\n"
                f"Released: {row['Released_Year']}  |  IMDb: {row['IMDB_Rating']}⭐\n\n"
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
    st.write("Enter a movie you love and press **Enter** to get five similar gems — complete with posters and key details.")

    def on_enter():
        title = st.session_state.get("movie_input", "").strip()
        if title:
            display_recommendations(title, df, cosine_sim, indices)

    st.text_input("Movie Title:", key="movie_input", on_change=on_enter)

############################################
# 6. Bootstrap
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
