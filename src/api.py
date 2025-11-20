from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import pandas as pd
from thefuzz import process

# Initialize FastAPI app
app = FastAPI(title="Movie Recommender API")

# Enables the browser to communicate with Python.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Global variables to hold the model data
cosine_sim = None
movies_df = None
indices = None

@app.on_event("startup")
def load_model():
    # this function runs once at startup to load the trained model
    global cosine_sim, movies_df, indices
    model_path = os.path.join("models", "movie_data.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Please train the model first.")
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        cosine_sim = data["similarity_matrix"]
        movies_df = data["dataframe"]

        # Create a reverse mapping of movie titles to indices
        indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

@app.get("/")
def home():
    return {"message": "Welcome to the Movie Recommender API!"}

@app.get("/recommend/{title}")
def recommend_movies(title: str):
    # Recommend movies based on the given title
    #try to find the exact title first
    if title in indices:
        idx = indices[title]
        movie_found = title
    else:
        # If not found, use fuzzy matching to find the closest title
        all_titles = movies_df['title'].tolist()
        match = process.extractOne(title, all_titles)
        best_match_name = match[0] 
        score = match[1]

        if score < 80:  # threshold for a good match
            raise HTTPException(status_code=404, detail="Movie title not found.")
        
        idx = indices[best_match_name]
        movie_found = best_match_name # use the best matched title
    

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores x[1] meaning the score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 5 most similar movies (excluding itself [0])
    sim_scores = sim_scores[1:6] 
    # Get this movies indices
    movie_indices = [i[0] for i in sim_scores]  
    recommendations = movies_df['title'].iloc[movie_indices].tolist()

    return {"movie": movie_found, "recommendations": recommendations}