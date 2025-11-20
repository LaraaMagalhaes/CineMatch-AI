from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import pandas as pd

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
    #verify if the title exists in the dataset
    if title not in indices:
        raise HTTPException(status_code=404, detail="Movie not found try the same title in english")
    
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores x[1] meaning the score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 5 most similar movies (excluding itself [0])
    sim_scores = sim_scores[1:6] 
    # Get this movies indices
    movie_indices = [i[0] for i in sim_scores]  
    recommendations = movies_df['title'].iloc[movie_indices].tolist()

    return {"movie": title, "recommendations": recommendations}