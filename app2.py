"""
Streamlit Movie Recommender — minimal pickle version
===================================================

Reads **movies.pkl** containing only these columns:

```
id, title, overview, tags
```

* `tags` is a stringified Python list (e.g. `"['crime', 'tarantino', 'cult']"`).
* Recommendations rely on TF‑IDF similarity of `overview + tags`.

Run with:
```bash
streamlit run app.py
```
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent / "data"
DATA_PATH      = DATA_DIR / "movies.pkl"
TOP_N = 10  # default number of recommendations

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_movies(pkl_path: Path) -> pd.DataFrame:
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found")

    df = pd.read_pickle(pkl_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Pickle must contain a pandas DataFrame")

    required = {"title", "overview", "tags"}
    if missing := required - set(df.columns):
        raise ValueError(f"DataFrame missing columns: {', '.join(missing)}")

    # Parse tags -> text
    df["tags_list"] = df["tags"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else str(x).split(",")
    )
    df["tags_text"] = df["tags_list"].apply(" ".join)

    # Combine overview + tags
    df["content"] = df["overview"].fillna("").astype(str) + " " + df["tags_text"]

    df["title_lower"] = df["title"].str.lower()
    return df.set_index("title_lower", drop=False)


@st.cache_resource(show_spinner=False)
def build_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["content"])
    return cosine_similarity(tfidf_matrix)


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def recommend(title: str, df: pd.DataFrame, sim_matrix: np.ndarray, top_n: int = TOP_N) -> List[str]:
    key = title.lower().strip()
    if key not in df.index:
        return []

    idx = df.index.get_loc(key)
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores[1:], key=lambda t: t[1], reverse=True)[:top_n]
    return df.iloc[[i for i, _ in scores]]["title"].tolist()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():  # pragma: no cover
    st.set_page_config("🎬 Movie Recommender", "🎬", layout="wide")
    st.title("🎬 Movie Recommender — minimal")

    try:
        df = load_movies(DATA_PATH)
    except Exception as err:
        st.error(f"Failed to load data — {err}")
        st.stop()

    sim_matrix = build_similarity_matrix(df)

    # user input
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        movie = st.text_input("Type a movie title:")
    with col_btn:
        run = st.button("Recommend", use_container_width=True)

    top_n = st.slider("Number of recommendations", 5, 20, TOP_N, 1)

    if run:
        if not movie:
            st.warning("Please enter a movie title first.")
            st.stop()

        recs = recommend(movie, df, sim_matrix, top_n=top_n)
        if not recs:
            st.info("Movie not found in data set.")
        else:
            st.subheader("You might also like:")
            for r in recs:
                st.write(f"• {r}")


if __name__ == "__main__":
    main()
