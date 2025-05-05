import os, json, re, stat, pathlib, base64, glob, ast
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path
import joblib

# ── NLTK Setup for Notebook-Style Recommender ────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def lemmatize_and_remove_stopwords(text: str):
    tokens = word_tokenize(text.lower())
    lems   = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]
    return [tok for tok in lems if tok not in stop_words]

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return float(inter/union) if union else 0.0

def merge_and_unique(row):
    tags = []
    for col in ['Genre','director','cast']:
        for item in row.get(col, "").split(','):
            item = item.strip()
            if item:
                tags.append(item)
    return list(set(tags))

def combined_similarity(idx, cos_sim_matrix, df):
    idx_pos = df.index.get_loc(idx)
    results = []
    for i in df.index:
        i_pos = df.index.get_loc(i)
        jac = jaccard_similarity(df.at[idx, 'tags'], df.at[i, 'tags'])
        cos = cos_sim_matrix[idx_pos, i_pos]
        results.append((i, 0.5*cos + 0.5*jac))
    return sorted(results, key=lambda x: x[1], reverse=True)

@st.cache_resource(show_spinner=False)
def load_nb_cosine_and_tags(df: pd.DataFrame):
    # Build tags column on movies.csv data
    df['tags'] = df.apply(merge_and_unique, axis=1)
    # TF-IDF on the Overview/overview text
    vectorizer_nb = TfidfVectorizer(tokenizer=lemmatize_and_remove_stopwords, max_features=5000)
    # Determine which overview column to use
    if 'Overview' in df.columns:
        texts = df['Overview']
    elif 'overview' in df.columns:
        texts = df['overview']
    else:
        texts = pd.Series([''] * len(df), index=df.index)
    tfidf_nb      = vectorizer_nb.fit_transform(texts.fillna(''))
    # Cosine similarity matrix
    cos_nb        = linear_kernel(tfidf_nb, tfidf_nb)
    return cos_nb

###############################################################################
# 0. ──────────────────────────────  CONSTANTS  ────────────────────────────── #
###############################################################################
USER_FILE = pathlib.Path.home() / ".movie_recs_secure" / "users.json"
USER_FILE.parent.mkdir(parents=True, exist_ok=True)
PEPPER = os.getenv("MOVIEREC_PEPPER", "").encode()

DATA_DIR   = Path(__file__).parent / "data"
MOVIES_CSV      = DATA_DIR / "movies.csv"
TMDB_CSV        = DATA_DIR / "TMDB_all_movies.csv"
VECTORIZER_FILE  = DATA_DIR / "tfidf_vectorizer.joblib"
INDEX_FILE       = DATA_DIR / "cosine_index.joblib"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

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


def verify_pw(password: str, hashed) -> bool:
    import bcrypt
    if not isinstance(hashed, str):
        return False
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
    col1, col_strip = st.columns(2)
    with col1:
        if st.button("Login"):
            if username in users and verify_pw(password, users[username]):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                safe_rerun()
            else:
                st.error("Invalid credentials")
    with col_strip:
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
@st.cache_data(show_spinner=False)
def load_dataframe(path: Path) -> pd.DataFrame:
    # Load CSV (movies.csv or TMDB_all_movies.csv)
    return pd.read_csv(path)

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
    # DataFrame for recommendation
    df_movies = load_dataframe(MOVIES_CSV)
    # Normalize movie key column for movies.csv
    if 'Series_Title' not in df_movies.columns:
        if 'title' in df_movies.columns:
            df_movies = df_movies.rename(columns={'title': 'Series_Title'})
        elif 'Title' in df_movies.columns:
            df_movies = df_movies.rename(columns={'Title': 'Series_Title'})
    # DataFrame for display
    df_tmdb = load_dataframe(TMDB_CSV)
    # Normalize movie key column for TMDB data
    if 'Series_Title' not in df_tmdb.columns:
        if 'title' in df_tmdb.columns:
            df_tmdb = df_tmdb.rename(columns={'title': 'Series_Title'})
        elif 'Title' in df_tmdb.columns:
            df_tmdb = df_tmdb.rename(columns={'Title': 'Series_Title'})
    # Compute similarity on movies.csv
    cos_nb = load_nb_cosine_and_tags(df_movies)
    return df_movies, cos_nb, df_tmdb


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
def recommender_ui(df_movies: pd.DataFrame, cos_sim_matrix, df_tmdb: pd.DataFrame):
    st.title("Movie Recommender")
    movie = st.text_input("Enter a movie title:")
    if st.button("Recommend"):
        if not movie:
            st.warning("Please enter a movie title to get recommendations.")
            return
        matches = df_movies[df_movies['Series_Title'].str.lower() == movie.lower()]
        if matches.empty:
            st.error(f"Movie '{movie}' not found in our dataset.")
            return
        sel_idx = matches.index[0]
        sims = combined_similarity(sel_idx, cos_sim_matrix, df_movies)[:6]
        rec_indices = [i for i,_ in sims if i != sel_idx][:5]
        for idx in rec_indices:
            title = df_movies.at[idx, 'Series_Title']
            tmdb_row = df_tmdb[df_tmdb['Series_Title'].str.lower() == title.lower()]
            row = tmdb_row.iloc[0] if not tmdb_row.empty else df_movies.loc[idx]
            cols = st.columns([1,3])
            with cols[0]:
                if 'poster_path' in row and pd.notna(row['poster_path']):
                    url = TMDB_IMAGE_BASE + row['poster_path']
                    st.image(url, use_container_width=True)
                elif 'Poster_Link' in row and pd.notna(row['Poster_Link']):
                    st.image(row['Poster_Link'], use_container_width=True)
            with cols[1]:
                st.subheader(row.get('Series_Title', title))
                rating = row.get('vote_average') or row.get('imdb_rating')
                votes = row.get('vote_count') or row.get('No_of_Votes')
                info = []
                if pd.notna(rating):
                    info.append(f"Rating: {rating}")
                if pd.notna(votes):
                    info.append(f"Votes: {int(votes)}")
                if info:
                    st.write(' · '.join(info))
                desc = row.get('overview') or row.get('Overview')
                if pd.notna(desc):
                    st.write(desc)
                rd = row.get('release_date')
                if pd.notna(rd):
                    st.write(f"Release Date: {rd}")
            st.markdown("---")

###############################################################################
# 8. ──────────────────────────────  MAIN  ────────────────────────────────── #
###############################################################################
def main():
    init_auth_state()
    if not st.session_state.get("logged_in", False):
        if not st.session_state.get("show_login", False):
            welcome_screen()
        else:
            login_ui()
    else:
        df_movies, cos_nb, df_tmdb = load_data()
        recommender_ui(df_movies, cos_nb, df_tmdb)

if __name__ == "__main__":
    main()

