# app.py
import os, json, re, getpass, stat, pathlib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path

###############################################################################
# 0. ──────────────────────────────  CONSTANTS  ────────────────────────────── #
###############################################################################
USER_FILE = pathlib.Path.home() / ".movie_recs_secure" / "users.json"
USER_FILE.parent.mkdir(parents=True, exist_ok=True)
PEPPER = os.getenv("MOVIEREC_PEPPER", "").encode()

###############################################################################
# 1. ────────────────────────────  USER STORAGE  ──────────────────────────── #
###############################################################################
def load_users() -> dict:
    if USER_FILE.exists():
        with open(USER_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_users(users: dict) -> None:
    tmp = USER_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    tmp.replace(USER_FILE)
    USER_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)

def hash_pw(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode() + PEPPER, bcrypt.gensalt()).decode()

def verify_pw(password: str, hashed: str) -> bool:
    import bcrypt
    try:
        return bcrypt.checkpw(password.encode() + PEPPER, hashed.encode())
    except ValueError:
        return False

###############################################################################
# 2. ────────────────────────  SESSION-STATE HELPERS  ─────────────────────── #
###############################################################################
def init_auth_state():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("users", load_users())
    st.session_state.setdefault("show_login", False)

def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

###############################################################################
# 3. ───────────────────────────  WELCOME UI  ────────────────────────────── #
###############################################################################
def welcome_screen():
    # Display logo and a centered, large orange Get Started button
    st.markdown("""
    <style>
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        margin: 0;
        padding: 0;
        background-color: #000;
    }
    .welcome-logo {
        max-width: 300px;
        margin-bottom: 40px;
    }
    .welcome-container .stButton > button {
        padding: 20px 60px !important;
        font-size: 28px !important;
        background-color: #FFA500 !important;
        color: #000 !important;
        border: none !important;
        border-radius: 10px !important;
        cursor: pointer;
    }
    .welcome-container .stButton > button:hover {
        background-color: #e69500 !important;
    }
    </style>
    <div class="welcome-container">
        <img src="/mnt/data/ChatGPT Image Apr 29, 2025, 11_13_51 AM.png" class="welcome-logo" />
    """, unsafe_allow_html=True)

    # Streamlit button inside the styled container
    if st.button("Get Started!", key="welcome_get_started"):
        st.session_state["show_login"] = True

    st.markdown("</div>", unsafe_allow_html=True)

###############################################################################
# 4. ─────────────────────────────  AUTH UI  ──────────────────────────────── #
###############################################################################
def strong_pwd(pwd: str) -> bool:
    return len(pwd) >= 8

def login_screen():
    st.markdown("""
    <style>
    .login-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background-image: url('https://example.com/login-bg.jpg');
        background-size: cover;
    }
    .login-box {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 40px;
        border-radius: 10px;
        width: 400px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
        color: #FAFAFA;
    }
    .login-box input {
        width:100%;padding:10px;margin-bottom:15px;border:none;border-radius:5px;
    }
    .login-box .stButton > button {
        width:100%;padding:10px;background-color:#1f77b4;color:white;border:none;border-radius:5px;font-size:16px;cursor:pointer;
    }
    .login-box .stButton > button:hover {
        background-color:#1b5fa7;
    }
    </style>
    <div class="login-container"><div class="login-box">
    <h2>Movie Recommender</h2><h3>Welcome</h3>
    """, unsafe_allow_html=True)

    mode = st.radio("", ["Log in", "Sign up", "Forgot password"], horizontal=True, key="auth_mode")
    users = st.session_state["users"]

    if mode == "Log in":
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Log in", key="do_login"):
            if username in users and verify_pw(password, users[username]["pw_hash"]):
                st.session_state["logged_in"] = True
                st.session_state["username"]  = username
                safe_rerun()
            else:
                st.error("Incorrect username or password.")

    elif mode == "Sign up":
        username = st.text_input("Choose a username", key="signup_user")
        email = st.text_input("E-mail", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pwd")
        confirm = st.text_input("Confirm password", type="password", key="signup_conf")
        if st.button("Create account", key="do_signup"):
            if username in users:
                st.error("Username already exists.")
            elif not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Invalid e-mail.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif not strong_pwd(password):
                st.error("Password must be at least 8 characters.")
            else:
                users[username] = {"email": email, "pw_hash": hash_pw(password), "likes": []}
                save_users(users)
                st.success("Account created – you’re now logged in!")
                st.session_state.update(logged_in=True, username=username)
                safe_rerun()

    else:  # Forgot password
        username = st.text_input("Username", key="reset_user")
        email = st.text_input("Registered e-mail", key="reset_email")
        new_pwd = st.text_input("New password", type="password", key="reset_new")
        confirm = st.text_input("Confirm new password", type="password", key="reset_conf")
        if st.button("Reset password", key="do_reset"):
            if username in users and users[username]["email"].lower() == email.lower():
                if new_pwd != confirm:
                    st.error("Passwords do not match.")
                elif not strong_pwd(new_pwd):
                    st.error("Password must be at least 8 characters.")
                else:
                    users[username]["pw_hash"] = hash_pw(new_pwd)
                    save_users(users)
                    st.success("Password reset – you can log in now.")
                safe_rerun()
            else:
                st.error("Username / e-mail mismatch.")

    st.markdown("</div></div>", unsafe_allow_html=True)

###############################################################################
# 4. ─────────────────────  MOVIE DATA & SIMILARITY  ──────────────────────── #
###############################################################################
def create_weighted_features(row):
    """Turn each movie’s numeric + categorical attributes into a token string
    that the TF‑IDF vectoriser can consume.  All numeric conversions are wrapped
    in try/except so bad strings (e.g. '') never crash the app."""
    def safe_int(val, divisor=1):
        try:
            return int(round(float(val) / divisor))
        except (ValueError, TypeError):
            return 0

    rating_tokens    = " rating"    * safe_int(row.get("IMDB_Rating", 0))
    metascore_tokens = " metascore" * safe_int(row.get("Meta_score", 0), divisor=10)

    genre_tokens    = (" " + row.get("Genre",    "")) * 3
    director_tokens = (" " + row.get("Director", ""))
    star_tokens     = (" " + row.get("Star1",    ""))

    return (rating_tokens + metascore_tokens + genre_tokens +
            director_tokens + star_tokens).strip()

@st.cache_resource(show_spinner=True)

def load_data():
    BASE = Path(__file__).parent 
    DATA_PATH = BASE / "data" / "imdb_top_1000.csv"
    df = pd.read_csv(DATA_PATH)
    df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce").fillna(0)
    df["Meta_score"]  = pd.to_numeric(df.get("Meta_score"), errors="coerce").fillna(0)

    df.fillna("", inplace=True)
    df["weighted"] = df.apply(create_weighted_features, axis=1)
    tfidf = TfidfVectorizer(stop_words="english")
    mat = tfidf.fit_transform(df["weighted"])
    cos = linear_kernel(mat, mat)
    idx = pd.Series(df.index, index=df["Series_Title"].str.lower()).drop_duplicates()
    return df, cos, idx

def recommend(title, df, cos, idx):
    t = title.lower().strip()
    if t not in idx:
        return None
    i = idx[t]
    scores = sorted(
        enumerate(cos[i]), key=lambda x: x[1], reverse=True
    )[1:6]  # top‑5
    return df.iloc[[s[0] for s in scores]].reset_index(drop=True)

###############################################################################
# 5. ─────────────────────────  RECOMMENDER UI  ───────────────────────────── #
###############################################################################
# ─────────────────────────────────────────────────────────────────────────────
# Add this helper *above* your recommender_ui definition:
def _trigger_recs():
    # Called when user types Enter in the text_input
    st.session_state["trigger_rec"] = True
    # Clear any old recs so we regenerate
    st.session_state.pop("last_recs", None)
# ─────────────────────────────────────────────────────────────────────────────

def recommender_ui(df, cos, idx):
    st.title("🎬 Movie Recommender")

    username = st.session_state["username"]
    users    = st.session_state["users"]

    # ── 1) SEARCH BOX: on_change fires when Enter is pressed ────────────────
    movie = st.text_input(
        "Enter a movie you love:",
        key="search_movie",
        on_change=_trigger_recs
    )

    # ── 2) MOVIE INFO CARD ──────────────────────────────────────────────────
    if movie:
        match = df[df["Series_Title"].str.lower() == movie.lower().strip()]
        if not match.empty:
            r = match.iloc[0]
            st.subheader("📖 Movie Info")
            c1, c2 = st.columns([1, 5])
            with c1:
                if r["Poster_Link"]:
                    st.image(r["Poster_Link"], width=120)
            with c2:
                stars = [r.get(f"Star{i}", "") for i in range(1,5) if r.get(f"Star{i}")]
                st.markdown(
                    f"**{r['Series_Title']}**\n\n"
                    f"Released: {r['Released_Year']} | IMDb {r['IMDB_Rating']}⭐\n\n"
                    f"Director: {r['Director']}\n\n"
                    f"Stars: {', '.join(stars)}\n\n"
                    f"Genre: {r['Genre']}\n\n"
                    f"**Overview:**  \n{r.get('Overview','No overview available.')}"
                )
                info_like_key = f"info_like_{r['Series_Title']}"
                if st.button("★ Like this movie", key=info_like_key):
                    likes = users[username]["likes"]
                    if r["Series_Title"] not in likes:
                        likes.append(r["Series_Title"])
                        save_users(users)
                        st.toast("Added to your likes!", icon="❤️")
                        safe_rerun()
        else:
            st.info(f"‘{movie}’ not found in our database.")

    st.markdown("---")

    # ── 3) GENERATE & STORE RECS ───────────────────────────────────────────
    # Fires on Recommend button OR on Enter (trigger_rec)
    if st.button("Recommend", key="recommend_btn") or st.session_state.get("trigger_rec"):
        st.session_state["trigger_rec"] = False
        recs = recommend(movie, df, cos, idx)
        st.session_state["last_recs"] = recs

    # ── 4) DISPLAY RECS PERSISTENTLY ──────────────────────────────────────
    recs = st.session_state.get("last_recs", None)
    if recs is not None:
        if recs.empty:
            st.warning(f"‘{movie}’ not found; can’t recommend similar titles.")
        else:
            st.success(f"Other movies you might like")
            for _, r in recs.iterrows():
                c1, c2 = st.columns([1, 5])
                with c1:
                    if r["Poster_Link"]:
                        st.image(r["Poster_Link"], width=110)
                with c2:
                    stars = [r.get(f"Star{i}", "") for i in range(1,5) if r.get(f"Star{i}")]
                    st.markdown(
                        f"**{r['Series_Title']}**\n\n"
                        f"Released: {r['Released_Year']} | IMDb {r['IMDB_Rating']}⭐\n\n"
                        f"Director: {r['Director']}\n\n"
                        f"Stars: {', '.join(stars)}\n\n"
                        f"Genre: {r['Genre']}\n\n"
                        f"**Overview:**  \n{r.get('Overview','No overview available.')}"
                    )
                    rec_like_key = f"rec_like_{r['Series_Title']}"
                    if st.button("★ Like", key=rec_like_key):
                        likes = users[username]["likes"]
                        if r["Series_Title"] not in likes:
                            likes.append(r["Series_Title"])
                            save_users(users)
                            st.toast("Added to your likes!", icon="❤️")
                            safe_rerun()

    # ── 5) SIDEBAR: IMMEDIATELY SHOW UPDATED LIKES ─────────────────────────
    st.sidebar.header(f"👤 {username}")
    likes = users[username]["likes"]
    if likes:
        st.sidebar.subheader("Your liked movies")
        # iterate over a copy so we can remove safely
        for title in likes.copy():
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            col1.write(f"• {title}")
            # “Remove” button
            if col2.button("✂️", key=f"unlike_{title}"):
                likes.remove(title)
                save_users(users)
                st.toast(f"Removed '{title}' from your likes.", icon="🗑️")
                safe_rerun()
    else:
        st.sidebar.info("No liked movies yet – show some ❤️!")

###############################################################################
# 6. ─────────────────────────────  MAIN  ─────────────────────────────────── #
###############################################################################
def main():
    init_auth_state()
    if not st.session_state["show_login"]:
        welcome_screen()
    elif not st.session_state["logged_in"]:
        login_screen()
    else:
        df, cos, idx = load_data()
        recommender_ui(df, cos, idx)

if __name__ == "__main__":
    main()
