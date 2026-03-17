# Paper Discovery (arXiv Searcher)

Paper Discovery is a tool designed to search, analyze, and group research papers from arXiv. It provides a Streamlit web interface for querying papers and applying machine learning techniques to cluster them by topic.

## Features

- **Search**: Query arXiv by keywords, categories, and date range.
- **Clustering**: Groups similar papers using Sentence Transformers, UMAP, HDBSCAN, and Agglomerative Clustering.
- **Topic Extraction**: Identifies representative key terms for each cluster using c-TF-IDF.
- **Visualization**: Displays clustered results in Plotly charts, showing the distribution of papers over time.


## Project Structure (`src/`)

- `streamlit_app.py`: Streamlit web interface.
- `arxiv_searcher.py`: Query building and data fetching from the arXiv API using `arxivql`.
- `exceptions.py`: Custom exceptions.
- `ml/`:
  - `preprocessing.py`: Text cleaning, normalization, lemmatization, and embedding generation.
  - `ml_utils.py`: Clustering implementations and evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).

## How to Run

### 1. Prerequisites
This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

### 2. Start the App
Run the following command from the root directory of the project to start the web interface:

```bash
uv run streamlit run src/streamlit_app.py
```

## Configuration & Logs
- **`src/config.json`**: Directory configuration for logging.
- **Logs**: Execution and error logs are saved to the directory specified in the configuration.