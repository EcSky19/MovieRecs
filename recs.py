import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import kagglehub

path = "/Users/ethancoskay/.cache/kagglehub/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows/versions/1"
csv_path = os.path.join(path, "imdb_top_1000.csv")

df = pd.read_csv(csv_path)

# Quick data inspection and cleaning
# Print the column names to confirm exactly how they appear in the file
print("Columns in dataset:", df.columns.tolist())

# Fill NaN with empty string to avoid errors
for col in ["Director", "Genre", "Star1", "Star2", "Star3", "Star4"]:
    df[col] = df[col].fillna("")

# Create a combined string:
df["combined_features"] = (
    df["Director"] + " " +
    df["Genre"] + " " +
    df["Star1"] + " " +
    df["Star2"] + " " +
    df["Star3"] + " " +
    df["Star4"]
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# For quick lookups, create a Series mapping titles to their DataFrame index
# Lowercase the title to make matching easier
df["Series_Title_lower"] = df["Series_Title"].str.lower()
indices = pd.Series(df.index, index=df["Series_Title_lower"]).drop_duplicates()

# Recommendation function
def recommend_movies(dataframe):
    """
    This function prompts the user for input, filters the dataframe,
    then sorts and prints the top results.
    """

    # Get user inputs
    user_genre = input("Enter a genre (e.g., Drama, Action): ").strip()
    user_director = input("Enter a director's name or partial name (e.g., Spielberg): ").strip()
    min_metascore_str = input("Enter a minimum Meta Score (0-100): ").strip()

    # Basic validation / default value
    try:
        min_metascore = int(min_metascore_str)
    except ValueError:
        print("Invalid Meta Score input. Defaulting to 0.")
        min_metascore = 0

    # Filter by genre, director, and metascore
    # Note: We use 'str.contains(...)' with case=False for case-insensitive matches.
    filtered = dataframe[
        (dataframe["Genre"].str.contains(user_genre, case=False)) &
        (dataframe["Director"].str.contains(user_director, case=False)) &
        (dataframe["Meta_score"] >= min_metascore)
    ]

    # Sort the results primarily by IMDB Rating, then by Meta Score
    filtered = filtered.sort_values(by=["IMDB_Rating", "Meta_score"], ascending=False)

    # Print results
    if filtered.empty:
        print("No matching movies found with your specified criteria.")
    else:
        print("\nTop recommended movies based on your criteria:\n")
        # Show top 10
        for i, row in filtered.head(10).iterrows():
            print(f"Title: {row['Series_Title']} | Year: {row['Released_Year']} | "
                  f"IMDB Rating: {row['IMDB_Rating']} | Meta Score: {row['Meta_score']} | "
                  f"Director: {row['Director']} | Genre: {row['Genre']}")
            
if __name__ == "__main__":
    recommend_movies(df)