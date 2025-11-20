import pickle
import os

from processing import load_dataset, clean_data, create_similarity_matrix

def main():
    # configurations
    input_csv = os.path.join("data", "tmdb_5000_movies.csv")
    output_folder = "models"
    output_model = os.path.join(output_folder, "movie_data.pkl")


    # Load dataset
    df_raw = load_dataset(input_csv)
    if df_raw is None:
        return

    # Clean data
    df_clean = clean_data(df_raw)

    # Treining the model (vetorizing and creating similarity matrix)
    cosine_sim = create_similarity_matrix(df_clean)

    # saving the model
    os.makedirs(output_folder, exist_ok=True)
    data = {
        "similarity_matrix": cosine_sim,
        "dataframe": df_clean[['title', 'id']]
    }

    with open(output_model, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()


