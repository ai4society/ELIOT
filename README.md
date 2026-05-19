# Eliot

Eliot is a Streamlit app for traceable exploration of fast-changing scientific literature. It retrieves arXiv papers from explicit query settings, clusters titles and abstracts into themes, labels those themes, and shows how they change over publication years.

Live app: https://ai4society-paper-searcher.streamlit.app

## Availability Notes

- The hosted app may sleep after a few idle hours because Streamlit Cloud can scale it to zero. If that happens, the first load may take a few minutes.
- Some searches may fail when arXiv rate-limits requests. Wait a few minutes and try the query again.

## Branch

This is the `deploy/streamlit-cloud` branch. It contains the app-only tree used for Streamlit Cloud deployment. Offline evaluation notebooks, datasets, and result artifacts live on `main`.

## What It Does

- Searches arXiv by keywords or phrases, category, date range, sort order, and maximum paper count.
- Clusters retrieved papers with SentenceTransformer embeddings, UMAP, and HDBSCAN or Agglomerative Clustering.
- Extracts representative cluster terms with c-TF-IDF.
- Shows temporal cluster distributions, clustering metrics, paper metadata, abstracts, and arXiv/PDF links.

## Run Locally

Requires Python 3.12 and `uv`.

```bash
uv sync
uv run streamlit run src/streamlit_app.py
```

The first run may download NLTK resources and the `all-MiniLM-L6-v2` sentence-transformer model.

## Deployment Notes

- Streamlit entry point: `src/streamlit_app.py`
- Deployment dependencies: `requirements.txt`
- Project metadata: `pyproject.toml`
- Logs are configured by `src/config.json`; `ARXIV_SEARCHER_BASE_DIR` can override the base directory used for logs.

## Project Layout

```text
src/
  streamlit_app.py        # Streamlit UI and interaction flow
  arxiv_searcher.py       # arXiv query construction and result parsing
  ml/preprocessing.py     # text cleaning, embeddings, UMAP, and c-TF-IDF terms
  ml/ml_utils.py          # clustering, optimal-k heuristic, and metrics
  styles.css              # Streamlit styling
```
