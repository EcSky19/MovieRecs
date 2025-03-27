import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def create_recommender(data_path):
    """
    Creates a simple content-based recommender system using TF-IDF on the 'overview' column.
    
    :param data_path: Path to a CSV file containing movie data with columns:
                      - 'title'
                      - 'overview'
    :return: A function `get_recommendations(movie_title, top_n=5)` that returns
             the top N recommendations for a given movie title.
    """

    # 1. Load your data
    df = pd.read_csv(data_path)
    
    # 2. Fill missing overviews with empty strings (if any)
    df['overview'] = df['overview'].fillna('')
    
    # 3. Create a TF-IDF matrix of the overviews
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    
    # 4. Compute the similarity using linear_kernel (cosine similarity)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # 5. Construct a reverse mapping from movie titles to indices for quick lookup
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()