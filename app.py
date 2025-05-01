import os, json, re, stat, pathlib, base64, glob, ast
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path
import joblib

###############################################################################
# 0. ──────────────────────────────  CONSTANTS  ────────────────────────────── #
###############################################################################
USER_FILE = pathlib.Path.home() / ".movie_recs_secure" / "users.json"
USER_FILE.parent.mkdir(parents=True, exist_ok=True)
PEPPER = os.getenv("MOVIEREC_PEPPER", "").encode()

DATA_DIR = Path(__file__).parent / "data"
PARQUET_FILE    = DATA_DIR / "tmdb_latest.parquet"
VECTORIZER_FILE = DATA_DIR / "tfidf_vectorizer.joblib"
INDEX_FILE      = DATA_DIR / "similarity_index.joblib"

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
    st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
    BASE = Path(__file__).parent
    logo_path = BASE / "data" / "logo.png"
    if logo_path.exists():
        encoded = base64.b64encode(logo_path.read_bytes()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{encoded}" width="300" style="margin-bottom:40px;"/>',
            unsafe_allow_html=True
        )
    else:
        st.error("Logo not found at /data/logo.png")
    if st.button("Get Started!", key="welcome_get_started"):
        st.session_state["show_login"] = True
    st.markdown('</div>', unsafe_allow_html=True)

###############################################################################
# 4. ─────────────────────────────  AUTH UI  ──────────────────────────────── #
###############################################################################
def login_ui():
    users = st.session_state["users"]
    st.subheader("Login or Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if username in users and verify_pw(password, users[username]):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                safe_rerun()
            else:
                st.error("Invalid credentials")
    with col2:
        if st.button("Sign Up"):
            if username and username not in users:
                users[username] = hash_pw(password)
                save_users(users)
                st.success("User created! Please log in.")
            else:
                st.error("Username taken or invalid.")

###############################################################################
# 5. ───────────────────────  DATA PROCESSING  &  SIMILARITY  ─────────────────── #
###############################################################################
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={
        "title": "Series_Title",
        "vote_average": "IMDB_Rating",
        "popularity": "popularity",
        "genres": "Genre",
        "overview": "Overview",
        "poster_path": "poster_path",
        "release_date": "release_date",
        "director": "director",
        "cast": "cast",
        "vote_count": "vote_count"
    }, inplace=True)
    df["Released_Year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["imdb_rating"] = pd.to_numeric(df.get("IMDB_Rating"), errors="coerce").fillna(0)
    df["popularity"]  = pd.to_numeric(df.get("popularity"), errors="coerce").fillna(0)
    df["vote_count"]  = pd.to_numeric(df.get("vote_count"), errors="coerce").fillna(0)
    def parse_genres(x):
        try:
            lst = ast.literal_eval(x) if isinstance(x, str) else x
            if isinstance(lst, list):
                return ", ".join(g.get("name", str(g)) for g in lst)
            return str(lst)
        except:
            return str(x)
    df["Genre"] = df["Genre"].apply(parse_genres).fillna("")
    for col in ["Genre","director","cast","Overview","poster_path"]:
        df[col] = df.get(col, "").fillna("").astype(str)
    df["Poster_Link"] = df["poster_path"].apply(lambda p: f"https://image.tmdb.org/t/p/w500{p}" if p else "")
    df["rating_tokens"]     = df["imdb_rating"].astype(int).apply(lambda n: " rating" * n)
    df["popularity_tokens"] = (df["popularity"].astype(int)//10).apply(lambda n: " popularity" * n)
    df["genre_tokens"]      = (" " + df["Genre"]).str.repeat(3)
    df["director_tokens"]   = " " + df["director"]
    df["star_tokens"]       = " " + df["cast"]
    df["weighted"] = (
        df["rating_tokens"] + df["popularity_tokens"] +
        df["genre_tokens"]   + df["director_tokens"]   + df["star_tokens"]
    ).str.strip()
    C = df["imdb_rating"].mean()
    m = df["vote_count"].quantile(0.90)
    df["weighted_rating"] = df.apply(
        lambda x: (x["vote_count"]/(x["vote_count"]+m) * x["imdb_rating"]) +
                  (m/(m+x["vote_count"]) * C), axis=1
    )
    return df

@st.cache_data(show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    if PARQUET_FILE.exists():
        return pd.read_parquet(PARQUET_FILE)
    csvs = sorted(DATA_DIR.glob("tmdb_movies*.csv"))
    if not csvs:
        st.error("No CSV found in data/.")
        st.stop()
    df = pd.read_csv(
        csvs[-1],
        usecols=["title","vote_average","popularity","genres",
                 "overview","poster_path","release_date",
                 "director","cast","vote_count"],
        low_memory=False
    )
    df = preprocess_df(df)
    df.to_parquet(PARQUET_FILE, index=False)
    return df

@st.cache_resource(show_spinner=False)
def load_tfidf_and_cosine(df: pd.DataFrame):
    if VECTORIZER_FILE.exists() and INDEX_FILE.exists():
        tfidf = joblib.load(VECTORIZER_FILE)
        cos   = joblib.load(INDEX_FILE)
    else:
        tfidf = TfidfVectorizer(stop_words="english")
        mat   = tfidf.fit_transform(df["weighted"])
        cos   = linear_kernel(mat, mat)
        joblib.dump(tfidf, VECTORIZER_FILE)
        joblib.dump(cos, INDEX_FILE)
    idx = pd.Series(df.index, index=df["Series_Title"].str.lower()).drop_duplicates()
    return cos, idx


def load_data():
    df  = load_dataframe()
    cos, idx = load_tfidf_and_cosine(df)
    return df, cos, idx

###############################################################################
# 6. ─────────────────────  HYBRID RECOMMENDATION  ───────────────────────── #
###############################################################################
def hybrid_recommend(title: str, df: pd.DataFrame, cos, idx, top_n:int=10, content_weight:float=0.5):
    key = title.lower()
    if key not in idx:
        return pd.DataFrame()
    i = idx[key]
    sims = sorted(enumerate(cos[i]), key=lambda x: x[1], reverse=True)[1:]
    wr = df["weighted_rating"]
    wr_norm = (wr - wr.min())/(wr.max()-wr.min()) if wr.max()!=wr.min() else wr
    hybrid = [(j, content_weight*sim + (1-content_weight)*wr_norm.iloc[j]) for j,sim in sims]
    hybrid = sorted(hybrid, key=lambda x: x[1], reverse=True)
    inds = [j for j,_ in hybrid[:top_n]]
    return df.iloc[inds]

###############################################################################
# 7. ─────────────────────────────  UI  ────────────────────────────── #
###############################################################################
def recommender_ui(df, cos, idx):
    st.title("Hybrid Movie Recommender")
    movie = st.selectbox("Select a movie:", sorted(df["Series_Title"].tolist()))
    num   = st.slider("How many recommendations?", 5, 20, 10)
    alpha = st.slider("Content vs. Popularity weight", 0.0, 1.0, 0.5)
    if st.button("Recommend"):
        recs = hybrid_recommend(movie, df, cos, idx, top_n=num, content_weight=alpha)
        for _, row in recs.iterrows():
            cols = st.columns([1,3])
            with cols[0]:
                if row["Poster_Link"]:
                    st.image(row["Poster_Link"], use_column_width=True)
            with cols[1]:
                st.subheader(row["Series_Title"])
                st.write(f"Rating: {row['imdb_rating']} · Votes: {row['vote_count']} · Score: {row['weighted_rating']:.2f}")
                st.write(row["Overview"])
            st.markdown("---")

# main remains unchanged
if __name__ == "__main__":
    init_auth_state()
    if not st.session_state["logged_in"]:
        if not st.session_state["show_login"]:
            welcome_screen()
        else:
            login_ui()
    else:
        df, cos, idx = load_data()
        recommender_ui(df, cos, idx)
