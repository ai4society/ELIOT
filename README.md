# Eliot

Eliot is a Streamlit app for traceable exploration of fast-changing scientific literature. It retrieves arXiv papers from explicit query settings, clusters titles and abstracts into themes, labels those themes, and shows how they change over publication years.

Live app: https://ai4society-paper-searcher.streamlit.app

## Availability Notes

- The hosted app may sleep after a few idle hours because Streamlit Cloud can scale it to zero. If that happens, the first load may take a few minutes.
- Some searches may fail when arXiv rate-limits requests. Wait a few minutes and try the query again.

## What It Does

- Searches arXiv by keywords or phrases, category, date range, and sort order.
- Clusters retrieved papers with SentenceTransformer embeddings, UMAP, and HDBSCAN or Agglomerative Clustering.
- Extracts representative cluster terms with c-TF-IDF.
- Shows temporal cluster distributions, clustering metrics, paper metadata, abstracts, and arXiv/PDF links.

## Branches

- `main`: application source plus offline evaluation notebooks, datasets, and result artifacts.
- `deploy/streamlit-cloud`: Streamlit Cloud deployment branch with the app-only tree and deployment-specific controls.

## Run Locally

Requires Python 3.12 and `uv`.

```bash
uv sync
uv run streamlit run src/streamlit_app.py
```

The first run may download NLTK resources and the `all-MiniLM-L6-v2` sentence-transformer model.

## Project Layout

```text
src/
  streamlit_app.py        # Streamlit UI and interaction flow
  arxiv_searcher.py       # arXiv query construction and result parsing
  ml/preprocessing.py     # text cleaning, embeddings, UMAP, and c-TF-IDF terms
  ml/ml_utils.py          # clustering, optimal-k heuristic, and metrics
  styles.css              # Streamlit styling
evaluation/               # offline experiments and result artifacts
```

## Configuration

- `src/config.json` sets the log directory.
- `ARXIV_SEARCHER_BASE_DIR` can override the base directory used for logs.
- `requirements.txt` is exported for deployment; `pyproject.toml` and `uv.lock` are the local dependency source of truth.

## Evaluation

The `evaluation/` directory contains offline experiments across arXiv domains, including configuration comparisons for document representation, dimensionality reduction, and clustering quality.
