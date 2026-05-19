<div align="center">

# Eliot

**Traceable exploration of fast-changing scientific literature**

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?style=for-the-badge)](https://ai4society-paper-searcher.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge)
![Data Source](https://img.shields.io/badge/Data-arXiv-b31b1b?style=for-the-badge)
![Framework](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=for-the-badge)

Eliot retrieves arXiv papers from explicit query settings, clusters titles and abstracts into themes, labels those themes, and shows how they evolve over publication years.

**Streamlit app:** https://ai4society-paper-searcher.streamlit.app

</div>

## Availability Notes

- The hosted app may sleep after a few idle hours because Streamlit Cloud can scale it to zero. If that happens, the first load may take a few minutes.
- Some searches may fail when arXiv rate-limits requests. Wait a few minutes and try the query again.

## What Eliot Helps You Do

| Task                     | How Eliot Supports It                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Build a traceable corpus | Search arXiv by keywords or phrases, category, date range, and sort order.                                   |
| Find topic structure     | Cluster retrieved papers with SentenceTransformer embeddings, UMAP, and HDBSCAN or Agglomerative Clustering. |
| Interpret clusters       | Extract representative terms with c-TF-IDF.                                                                  |
| Inspect change over time | Plot cluster membership by publication year with hoverable paper details.                                    |
| Audit the evidence       | Browse paper cards with metadata, abstracts, arXiv links, and PDF links.                                     |

## Workflow

```text
keywords + filters
  -> arXiv retrieval
  -> title/abstract preprocessing
  -> MiniLM embeddings
  -> UMAP reduction
  -> clustering
  -> c-TF-IDF labels
  -> temporal visualization + paper inspection
```

## Run Locally

Requires Python 3.12 and `uv`.

```bash
uv sync
uv run streamlit run src/streamlit_app.py
```

The first run may download NLTK resources and the `all-MiniLM-L6-v2` sentence-transformer model.

## Repository Layout

```text
src/
  streamlit_app.py        # Streamlit UI and interaction flow
  arxiv_searcher.py       # arXiv query construction and result parsing
  ml/preprocessing.py     # text cleaning, embeddings, UMAP, and c-TF-IDF terms
  ml/ml_utils.py          # clustering, optimal-k heuristic, and metrics
  styles.css              # Streamlit styling

evaluation/
  datasets/               # offline arXiv-domain datasets
  notebook/               # evaluation and analysis notebooks
  results/                # metrics, rankings, samples, and visual artifacts
```

## Branches

| Branch                   | Purpose                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `main`                   | Application source plus offline evaluation notebooks, datasets, and result artifacts. |
| `deploy/streamlit-cloud` | App-only tree used for Streamlit Cloud deployment.                                    |

## Configuration

- `src/config.json` sets the log directory.
- `ARXIV_SEARCHER_BASE_DIR` can override the base directory used for logs.
- `pyproject.toml` and `uv.lock` are the local dependency source of truth.
- `requirements.txt` is exported for Streamlit Cloud deployment.

## Evaluation

The `evaluation/` directory contains offline experiments across arXiv domains, including configuration comparisons for document representation, dimensionality reduction, and clustering quality.

## Scope

Eliot is an exploratory aid. Cluster assignments, labels, and metrics help structure a literature search, but they should be interpreted alongside the underlying papers.
