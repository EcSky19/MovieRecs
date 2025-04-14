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