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
    
    def get_recommendations(movie_title, top_n=5):
        """
        Given a movie title, returns the top_n most similar movie titles
        based on the TF-IDF similarity of their overviews.
        """
        # Check if the movie_title exists in our data
        if movie_title not in indices:
            return [f"'{movie_title}' not found in dataset."]
        
        # Get the index of the movie that matches the title
        idx = indices[movie_title]
        
        # Get similarity scores for all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies by similarity score (descending), skipping the first match (itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Grab the top_n from the sorted list (starting from index 1, since 0 is the movie itself)
        sim_scores = sim_scores[1: top_n+1]
        
        # Get the movie indices of those top_n
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the titles of the top_n most similar
        return df['title'].iloc[movie_indices].tolist()
    
    return get_recommendations