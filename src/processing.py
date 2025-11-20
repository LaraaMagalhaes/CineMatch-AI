import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_dataset(file_path):
    # Load dataset from a CSV file.
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return None


def clean_data(df):
    # Clean the dataset by filling NaN values.
    df['overview'] = df['overview'].fillna('')
    return df


def create_similarity_matrix(df):
    # receive a dataframe with movie overviews, remove comum words, 
    # transform text in numbers and return the cosine similarity matrix.
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return cosine_sim
