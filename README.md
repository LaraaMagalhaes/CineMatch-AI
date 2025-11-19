# 🎬 What to Watch Next? (Content-Based Movie Recommender)

A robust machine learning application that recommends movies based on plot similarity. Unlike collaborative filtering (which relies on user history), this engine uses Natural Language Processing (NLP) and Linear Algebra to understand the content of the movies themselves, mapping them into a high-dimensional vector space.

## 🧠 The Logic (How it Works)

This project implements a Content-Based Filtering approach. The core idea is that if a user likes a movie with specific characteristics (keywords, genre, mood), they will likely enjoy other movies with a similar "mathematical signature."

### The Pipeline

- **Data Ingestion:** We use the standard TMDB 5000 Movie Dataset.
- **Text Preprocessing:** Movie plots/synopses are cleaned (stop-word removal, stemming) to remove noise.
- **Vectorization (TF-IDF):** Converts text into numerical vectors. Rare, descriptive words (like "lightsaber") carry more weight than common ones.
- **Similarity Calculation:** Computes Cosine Similarity between the requested movie vector and all other 4,999 movie vectors.
- **High Cosine Score (~1.0):** Plots are almost identical.  
- **Low Cosine Score (~0.0):** Plots are unrelated.

## 🏗 Architecture

The project is designed with a clear separation between **Model Training (Offline)** and **Model Serving (Online)** to ensure high performance and low latency.

```bash
graph LR
    A[Raw Data (CSV)] --> B(Preprocessing Script);
    B --> C{TF-IDF Vectorizer};
    C --> D[Serialized Model (.pkl)];
    D --> E[FastAPI Backend];
    F[Frontend Client] -- "GET /recommend/Batman" --> E;
    E -- "JSON Response" --> F;
```

### Offline Phase (Data Engineering):

A Python script processes the data and trains the vectorizer. The resulting matrix is serialized (pickled) to disk.

### Online Phase (API):

A FastAPI server loads the pre-trained model into memory on startup. When a request comes in, it performs a fast look-up, eliminating the need to re-train or re-process data for every user.

## 🛠 Tech Stack

### Machine Learning & Data
- Python: Core language.
- Pandas: Data manipulation and cleaning.
- Scikit-Learn: Implementation of TF-IDF and Cosine Similarity metrics.
- Pickle: Object serialization for model persistence.

### Backend (API)
- FastAPI: A modern, high-performance web framework for building APIs.  
- Uvicorn: ASGI server.

### Frontend
- HTML5 / CSS3: Structure and styling.  
- JavaScript: Handles asynchronous requests (fetch) to the Python backend and DOM manipulation to render results.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+  
- npm (optional)

### Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/what-to-watch-next.git
cd what-to-watch-next
```

Install dependencies:
```bash
pip install pandas scikit-learn fastapi uvicorn
```

Train the Model (Offline Step): Run the processing script to generate the similarity matrix.

```bash
python src/train_model.py
```

This will create a `model_data.pkl` file.

### Start the API Server

```bash
uvicorn src.api:app --reload
```

The API will be running at `http://127.0.0.1:8000`.

### Launch the Frontend

Open:

```
frontend/index.html
```

## 👨‍💻 Author

Anna Lara Magalhães Monteiro Vieira  
Computer Science Student @Universidade de Fortaleza  
Aspiring AI Engineer  
www.linkedin.com/in/laraamagalhaes
