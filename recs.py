import os
import pandas as pd

import kagglehub

# Download latest version
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")

print("Path to dataset files:", path)

path = "YOUR_DOWNLOADED_DATASET_FOLDER"  
csv_path = os.path.join(path, "IMDB Top 1000.csv")

df = pd.read_csv(csv_path)

# Quick data inspection and cleaning
# Print the column names to confirm exactly how they appear in the file
print("Columns in dataset:", df.columns.tolist())

df.dropna(subset=["Genre", "Director", "Meta_score", "IMDB_Rating"], inplace=True)

# Convert Meta_score to numeric if it's not already
df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
df.dropna(subset=["Meta_score"], inplace=True)

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