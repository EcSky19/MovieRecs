# Movie Recommender

A content-based movie recommender system that uses TF-IDF vectorization with custom weighting for numeric and text features (IMDB Rating, Meta Score, Genre, and Director). This project also includes a Streamlit-based frontend for a simple web interface.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features and Weights](#features-and-weights)  
3. [Setup and Installation](#setup-and-installation)  
4. [Dataset](#dataset)  
5. [Running the Recommender (CLI)](#running-the-recommender-cli)  
6. [Running the Streamlit App](#running-the-streamlit-app)  
7. [Customization](#customization)  
8. [License](#license)

---

## Project Overview

This project demonstrates a **content-based** recommender system that reads a dataset of top 1000 IMDB movies, creates “weighted” textual features, and uses TF-IDF plus cosine similarity to find and recommend similar movies.

**Core steps**:
1. **Data Loading**: Reads the CSV containing movie metadata (titles, genres, directors, ratings, metascores, etc.).  
2. **Feature Weighting**: Important numeric attributes (e.g., IMDB Rating, Meta Score) are converted to repeated tokens to give them heavier weight. Genre and Director are also repeated a certain number of times to reflect their relative importance.  
3. **TF-IDF Vectorization**: Transforms the weighted text into a numerical vector.  
4. **Similarity**: A cosine similarity matrix is computed for all movies.  
5. **Recommendation**: Given a movie title, the system retrieves the top 10 similar movies.

---

## Features and Weights

1. **IMDB Rating** (0–10) → repeated token counts equal to the rating (e.g., rating=8 → “rating” repeated 8 times).  
2. **Meta Score** (0–100) → repeated token counts equal to (meta_score ÷ 10).  
3. **Genre** → repeated 3× to give moderate-high weight.  
4. **Director** → repeated 1× to give minimal weight.  

These multipliers can be freely adjusted in code to emphasize or de-emphasize features.

---

## Setup and Installation

1. **Clone or Download**: Copy this repo to your local machine.
2. **Install Dependencies** (ideally in a virtual environment):
   ```bash
   pip install pandas scikit-learn streamlit kagglehub
