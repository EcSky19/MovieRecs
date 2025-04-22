# app.py
import os, json, re, bcrypt, getpass, stat, pathlib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

###############################################################################
# 0. ──────────────────────────────  CONSTANTS  ────────────────────────────── #
###############################################################################
USER_FILE = pathlib.Path.home() / ".movie_recs_secure" / "users.json"
USER_FILE.parent.mkdir(parents=True, exist_ok=True)
PEPPER = os.getenv("MOVIEREC_PEPPER", "").encode()

###############################################################################
# 1. ────────────────────────────  USER STORAGE  ──────────────────────────── #
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
    USER_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

def hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode() + PEPPER, bcrypt.gensalt()).decode()

def verify_pw(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode() + PEPPER, hashed.encode())
    except ValueError:
        return False

###############################################################################
# 2. ────────────────────────  SESSION‑STATE HELPERS  ─────────────────────── #
###############################################################################
def init_auth_state():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("users", load_users())

def safe_rerun():
    # Streamlit < 1.31 lacks experimental_rerun
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

###############################################################################
# 3. ─────────────────────────────  AUTH UI  ──────────────────────────────── #
###############################################################################
def strong_pwd(pwd: str) -> bool:
    return len(pwd) >= 8

def login_screen():
    mode = st.radio(
        "Choose an option",
        ["Log in", "Sign up", "Forgot password"],
        horizontal=True,
        key="auth_mode",
    )
    users = st.session_state["users"]

    if mode == "Log in":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log in"):
            if username in users and verify_pw(password, users[username]["pw_hash"]):
                st.session_state.update(logged_in=True, username=username)
                safe_rerun()
            else:
                st.error("Incorrect username or password.")

    elif mode == "Sign up":
        username = st.text_input("Choose a username")
        email = st.text_input("E‑mail")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        if st.button("Create account"):
            if username in users:
                st.error("Username already exists.")
            elif not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Invalid e‑mail.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif not strong_pwd(password):
                st.error("Password must be at least 8 characters.")
            else:
                users[username] = {
                    "email": email,
                    "pw_hash": hash_pw(password),
                    "likes": [],
                }
                save_users(users)
                st.success("Account created – you’re now logged in!")
                st.session_state.update(logged_in=True, username=username)
                safe_rerun()

    else:  # Forgot password
        username = st.text_input("Username")
        email = st.text_input("Registered e‑mail")
        new_pwd = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        if st.button("Reset password"):
            if (
                username in users
                and users[username]["email"].lower() == email.lower()
            ):
                if new_pwd != confirm:
                    st.error("Passwords do not match.")
                elif not strong_pwd(new_pwd):
                    st.error("Password must be at least 8 characters.")
                else:
                    users[username]["pw_hash"] = hash_pw(new_pwd)
                    save_users(users)
                    st.success("Password reset – you can log in now.")
            else:
                st.error("Username / e‑mail mismatch.")

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
    DATA_PATH = (
        "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/"
        "imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1/imdb_top_1000.csv"
    )
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
def recommender_ui(df, cos, idx):
    st.title("🎬 Movie Recommender")

    movie = st.text_input("Enter a movie you love:", key="search_movie")
    username = st.session_state["username"]
    users    = st.session_state["users"]

    # ── 1) Movie Info Card ─────────────────────────────────────────
    if movie:
        # find exact match (case‑insensitive)
        match = df[df["Series_Title"].str.lower() == movie.lower().strip()]
        if not match.empty:
            r = match.iloc[0]
            st.subheader("📖 Movie Info")
            col1, col2 = st.columns([1, 5])
            with col1:
                if r["Poster_Link"]:
                    st.image(r["Poster_Link"], width=120)
            with col2:
                # details + overview
                stars = [
                    r.get(f"Star{i}", "")
                    for i in range(1, 5)
                    if r.get(f"Star{i}", "")
                ]
                st.markdown(
                    f"**{r['Series_Title']}**\n\n"
                    f"Released: {r['Released_Year']} | IMDb {r['IMDB_Rating']}⭐\n\n"
                    f"Director: {r['Director']}\n\n"
                    f"Stars: {', '.join(stars)}\n\n"
                    f"Genre: {r['Genre']}\n\n"
                    f"**Overview:**  \n{r.get('Overview','No overview available.')}"
                )
                # own like‑button for the info card
                info_like_key = f"info_like_{r['Series_Title']}"
                if st.button("★ Like this movie", key=info_like_key):
                    if r["Series_Title"] not in users[username]["likes"]:
                        users[username]["likes"].append(r["Series_Title"])
                        save_users(users)
                        st.toast("Added to your likes!", icon="❤️")
        else:
            st.info(f"‘{movie}’ not found in our database.")

    st.markdown("---")  # separator between info and recs

    # ── 2) Recommendation Section ───────────────────────────────────
    if st.button("Recommend", key="recommend_btn"):
        recs = recommend(movie, df, cos, idx)
        if recs is None:
            st.warning(f"‘{movie}’ not found; can’t generate recommendations.")
        else:
            st.success(f"Movies similar to ‘{movie}’")
            for _, r in recs.iterrows():
                c1, c2 = st.columns([1, 5])
                with c1:
                    if r["Poster_Link"]:
                        st.image(r["Poster_Link"], width=110)
                with c2:
                    stars = [
                        r.get(f"Star{i}", "")
                        for i in range(1, 5)
                        if r.get(f"Star{i}", "")
                    ]
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
                        if r["Series_Title"] not in users[username]["likes"]:
                            users[username]["likes"].append(r["Series_Title"])
                            save_users(users)
                            st.toast("Added to your likes!", icon="❤️")

    # ── 3) Sidebar: show updated likes immediately ─────────────────
    st.sidebar.header(f"👤 {username}")
    liked = users[username]["likes"]
    if liked:
        st.sidebar.subheader("Your liked movies")
        for title in liked:
            st.sidebar.write(f"• {title}")
    else:
        st.sidebar.info("No liked movies yet – show some ❤️!")


###############################################################################
# 6. ─────────────────────────────  MAIN  ─────────────────────────────────── #
###############################################################################
def main():
    init_auth_state()
    if not st.session_state["logged_in"]:
        login_screen()
        return

    df, cos, idx = load_data()
    recommender_ui(df, cos, idx)

# capture Enter key on movie field
def _set_enter_hotkey():
    if not st.runtime.exists():
        return
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    if ctx:
        st.session_state.setdefault("trigger_rec", False)
        if ctx.request_rerun_data.widget_states.get("MovieRecommender") is not None:
            st.session_state["trigger_rec"] = True

if __name__ == "__main__":
    main()
