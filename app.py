import os
import json
import stat
import re
import pandas as pd
import streamlit as st
from passlib.hash import bcrypt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

############################################
# 0. Security Helpers
############################################

# Location of the on‑disk credential store (outside web root)
APP_DIR = os.path.expanduser("~/.movie_recs_secure")
USERS_FILE = os.path.join(APP_DIR, "users.json")

# Optional PEPPER (set an env‑var in production for extra security)
PEPPER = os.environ.get("MOVIEREC_PEPPER", "")

# Username policy: 3‑32 chars, alphanumerics, underscores, dashes
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{3,32}$")


def ensure_storage_dir() -> None:
    os.makedirs(APP_DIR, mode=0o700, exist_ok=True)


def secure_save(path: str, data: dict) -> None:
    """Write JSON with 0600 permissions (owner‑read/write only)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf‑8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600


############################################
# 1. Password Hashing
############################################

def hash_pw(password: str) -> str:
    """Hash a password using bcrypt + optional pepper."""
    return bcrypt.using(rounds=12).hash(password + PEPPER)


def verify_pw(password: str, hashed: str) -> bool:
    try:
        return bcrypt.verify(password + PEPPER, hashed)
    except Exception:
        return False


############################################
# 2. Persistent User Store
############################################

def load_user_db():
    ensure_storage_dir()
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf‑8") as f:
                return json.load(f)
        except Exception:
            pass  # Corrupt file — fall through to default
    # First‑run default admin account (password = admin123 — advise changing)
    default_users = {"admin": hash_pw("admin123")}
    secure_save(USERS_FILE, default_users)
    return default_users


def save_user_db(users):
    try:
        secure_save(USERS_FILE, users)
    except Exception:
        st.warning("⚠️ Failed to save credential store; new accounts won't persist.")


############################################
# 3. Auth Handler (log in / sign up)
############################################

def init_auth_state():
    st.session_state.setdefault("users", load_user_db())
    st.session_state.setdefault("logged_in", False)


def login_screen():
    st.title("Secure Movie Recommender — Sign In / Sign Up")
    mode = st.radio("Choose an option", ["Log in", "Sign up"], horizontal=True, key="auth_mode")

    if mode == "Log in":
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log in", key="login_btn"):
            users = st.session_state["users"]
            if username in users and verify_pw(password, users[username]):
                st.session_state["logged_in"] = True
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password.")

    else:  # Sign up
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Create account", key="signup_btn"):
            if not USERNAME_PATTERN.fullmatch(new_user or ""):
                st.error("Username must be 3‑32 chars: letters, numbers, _ or -")
            elif len(new_pass) < 8:
                st.error("Password must be at least 8 characters long.")
            elif new_user in st.session_state["users"]:
                st.error("Username already exists. Pick another.")
            else:
                st.session_state["users"][new_user] = hash_pw(new_pass)
                save_user_db(st.session_state["users"])
                st.success("Account created! You are now logged in.")
                st.session_state["logged_in"] = True
                st.experimental_rerun()


############################################
# 4. Movie Recommender (unchanged logic)
############################################

# --- Weighting & similarity helpers (same as before) ---

def create_weighted_features(row):
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
    director_tokens = " " + row["Director"]
    star_tokens = "".join(" " + row[c] for c in ["Star1", "Star2", "Star3", "Star4"])
    return (rating_tokens + metascore_tokens + genre_tokens + director_tokens + star_tokens).strip()


def load_data():
    DATA_PATH = (
        "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/"
        "imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
    )
    csv_path = os.path.join(DATA_PATH, "imdb_top_1000.csv")
    df = pd.read_csv(csv_path)
    for col in [
        "Series_Title","Released_Year","Genre","IMDB_Rating","Director",
        "Star1","Star2","Star3","Star4","Poster_Link","Meta_score","Overview"]:
        df[col] = df.get(col, "").fillna("")
    df["weighted_features"] = df.apply(create_weighted_features, axis=1)
    tfidf = TfidfVectorizer(stop_words="english")
    cosine_sim = linear_kernel(tfidf.fit_transform(df["weighted_features"]), tfidf.fit_transform(df["weighted_features"]))
    df["Series_Title_lower"] = df["Series_Title"].str.lower()
    indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()
    return df, cosine_sim, indices


def get_recommendations(title, df, cosine_sim, indices):
    t = title.lower().strip()
    if t not in indices:
        return None
    idx = indices[t]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:6]
    return df.iloc[[i[0] for i in sim_scores]][[
        "Series_Title","Released_Year","Director","Star1","Star2","Star3","Star4",
        "IMDB_Rating","Genre","Overview","Poster_Link"
    ]].reset_index(drop=True)


def display_recs(title, df, cosine_sim, indices):
    recs = get_recommendations(title, df, cosine_sim, indices)
    if recs is None:
        st.warning(f"'{title}' not found.")
        return
    st.success(f"Movies similar to '{title}':")
    for _, row in recs.iterrows():
        col_img, col_txt = st.columns([1,5])
        with col_img:
            if row["Poster_Link"]:
                st.image(row["Poster_Link"], width=120)
        with col_txt:
            actors = ", ".join(filter(None, [row[c] for c in ["Star1","Star2","Star3","Star4"]])) or "N/A"
            overview = row["Overview"][:480].rstrip() + ("…" if len(row["Overview"])>480 else "")
            st.markdown(
                f"**{row['Series_Title']}**\n\n"
                f"Released: {row['Released_Year']} | IMDb: {row['IMDB_Rating']}⭐\n\n"
                f"Director: {row['Director']}\n\n"
                f"Actors: {actors}\n\n"
                f"Genre: {row['Genre']}\n\n"
                f"**Overview:** {overview}"
            )


def recommender_ui(df, cosine_sim, indices):
    st.title("🎬 Secure Movie Recommender")
    st.write("Type a movie you love and press **Enter** to see five similar picks.")
    def on_enter():
        t = st.session_state.get("movie_query", "").strip()
        if t:
            display_recs(t, df, cosine_sim, indices)
    st.text_input("Movie Title", key="movie_query", on_change=on_enter)


############################################
# 5. App Entrypoint
############################################

def run():
    init_auth_state()
    if not st.session_state["logged_in"]:
        login_screen()
        return
    df, cosine_sim, indices = load_data()
    recommender_ui(df, cosine_sim, indices)


if __name__ == "__main__":
    run()
