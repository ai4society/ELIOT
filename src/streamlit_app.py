import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from arxiv_searcher import ARXIV_CATEGORIES, Paper, search
from ml.ml_utils import MIN_PAPERS_FOR_CLUSTERING, MAX_NUMBER_OF_CLUSTERS, get_optimal_k, get_papers_kmeans, get_paper_clusters_hdbscan
from exceptions import (
    ArxivSearcherError,
    InvalidKeywordError,
    InvalidDateRangeError,
    TooManyKeywordsError,
    ArxivFetchingError
)


@st.cache_data(ttl=timedelta(hours=1), max_entries=1000, show_spinner=False)
def search_papers(
    keywords: str, start_date: date, end_date: date, sort_opt: str, category_option: str
) -> List[Paper]:
    import pandas as pd
    import ast

    # Avoid calling arxiv api for repetitive tests
    # df = pd.read_csv(
    #     "../evaluation/datasets/new/papers_20230218-20260218_trustworthy_AI.csv",
    #     parse_dates=["published", "updated"]
    # )

    # records = df.to_dict("records")

    # for r in records:
    #     if isinstance(r["authors"], str):
    #         r["authors"] = ast.literal_eval(r["authors"])

    #     if isinstance(r["categories"], str):
    #         r["categories"] = ast.literal_eval(r["categories"])

    # return [Paper(**r) for r in records][0:200]

    return search(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_opt,
        category=category_option,
    )


def is_noise_cluster(cid: int) -> bool:
    """Helper to determine if a cluster ID represents the noise cluster."""
    method = st.session_state.get("clustering_method", "Auto-detect clusters")
    return cid == 1 and method == "Auto-detect clusters"


def create_cluster_viz(ordered_papers: List[Paper], all_labels: List[str], metrics: dict):
    # Use Year and Cluster for visualization
    years = [p.published.year for p in ordered_papers]
    cluster_nums = [int(label) for label in all_labels]
    K = len(set(all_labels))

    # Add jitter to avoid overlapping points since both axes are discrete
    np.random.seed(42)
    jitter_x = np.random.uniform(-0.3, 0.3, size=len(ordered_papers))
    jitter_y = np.random.uniform(-0.3, 0.3, size=len(ordered_papers))

    df_viz = pd.DataFrame({
        "Year": years,
        "Cluster": [HDBSCAN_NOISE_CLUSTER_NAME if is_noise_cluster(int(label)) else f"Cluster {label}" for label in all_labels],
        "x": np.array(years) + jitter_x,
        "y": np.array(cluster_nums) + jitter_y,
        "Title": [p.title for p in ordered_papers],
        "Date": [p.published.strftime("%Y-%m-%d") for p in ordered_papers]
    })

    fig = px.scatter(
        df_viz, 
        x="x", y="y", 
        color="Cluster", 
        color_discrete_map=color_map,
        hover_data={
            "Title": True, 
            "Date": True,
            "Cluster": True,
            "x": False, 
            "y": False
        },
        title="Paper Clusters"
    )
    
    # Format axes to show original categories (Years and Cluster IDs)
    fig.update_layout(
        # Place legend outside the plotting area
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top"
        ),
        xaxis=dict(
            title="Year",
            tickmode='linear',
            dtick=1,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            title="Cluster ID",
            tickmode='array',
            tickvals=sorted(list(set(cluster_nums))),
            ticktext=[HDBSCAN_NOISE_CLUSTER_NAME if is_noise_cluster(c) else f"Cluster {c}" for c in sorted(list(set(cluster_nums)))],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="closest"
    )

    fig.add_annotation(
        x=1.02,
        y=0,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text=(
            "Metrics:<br>"
            f"     <b>Davies–Bouldin</b>: {metrics['DBI']:.2f}<br>"
            f"     <b>Calinski–Harabasz</b>: {metrics['CHI']:.2f}<br>"
            f"     <b>Silhouette</b>: {metrics['SIL']:.3f}"
        ),
    )

    fig.update_layout(margin=dict(r=160))
    base_height = 500
    # Adjust plot height linearly after 10 clusters
    height = base_height + max(0, K - 10) * 30
    height = max(500, min(height, 850))
    return st.plotly_chart(fig, height=height)


def build_cluster_color_map(cluster_ids: List[str]):
    palette = px.colors.qualitative.Plotly
    color_map = {}
    for i, cid in enumerate(cluster_ids):
        if cid == 0:
            color_map["All Papers"] = palette[i % len(palette)]
        elif is_noise_cluster(cid):
            color_map[HDBSCAN_NOISE_CLUSTER_NAME] = palette[i % len(palette)]
        else:
            color_map[f"Cluster {cid}"] = palette[i % len(palette)]
    return color_map


def get_default_session_state():
    return {
        "papers": [], 
        "searched": False, 
        "clusters": {}, 
        "top_words": {}, 
        "metrics": {}, 
        "optimal_k": 1
    }


st.set_page_config(
    page_title="Paper Explorer", page_icon="📚", layout="centered"
)

css_path = Path(__file__).parent / "styles.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# header link
st.markdown(
    '<div class="header-bar"><a href="https://ai4society.github.io/" target="_blank" class="header-link">AI4Society</a></div>',
    unsafe_allow_html=True,
)

ORDER_BY_OPTIONS = {
    "Relevance": "relevance",
    "Submitted Date": "submitted",
}

DEFAULT_KEYWORDS = "large language models, multi-agent systems"
HDBSCAN_NOISE_CLUSTER_NAME = "Cluster 1"


# Initialize session state
if "search_results" not in st.session_state:
    st.session_state["search_results"] = get_default_session_state()

st.title("Paper Discovery", anchor=False)

st.markdown(
    '<div class="page-subtitle">Discover and explore research papers across the arXiv using keyword-based search</div>',
    unsafe_allow_html=True,
)

col_keywords, col_category = st.columns([2.5, 1])

with col_keywords:
    keywords = st.text_input(
        "Search Keywords or Phrases",
        placeholder=f"e.g., {DEFAULT_KEYWORDS} ...",
        help=(
            "Enter one or more keywords or phrases separated by commas. "
            "Results include only papers where all keywords are present. "
            f"If left empty, the default keywords - {DEFAULT_KEYWORDS} - will be used."
        ),
    )

with col_category:
    category_option = st.selectbox(
        "Category",
        ARXIV_CATEGORIES.keys(),
        index=0,
        help="Focus your search on a specific research field",
    )

col_start_date, col_end_date, col_order_by, col_search_bt = st.columns(
    [1, 1, 1.5, 1], vertical_alignment="center"
)

with col_start_date:
    # Default -> Starts from last year
    start_date = st.date_input(
        "Start Date",
        value=date(date.today().year - 1, date.today().month, date.today().day),
    )

with col_end_date:
    end_date = st.date_input("End Date", value="today")

with col_order_by:
    sort_option = st.selectbox(
        "Order By", ORDER_BY_OPTIONS.keys(), help="Order applied to arXiv search"
    )

with col_search_bt:
    # Keeps button's vertical alignment
    st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
    search_button = st.button("Search")

# On click
if search_button:
    # When making a new search, reset the session state
    st.session_state["clustering_method"] = "Auto-detect clusters"
    st.session_state["search_results"] = get_default_session_state()

    with st.spinner("🔎 Searching papers..."):
        try:
            if not keywords:
                keywords = DEFAULT_KEYWORDS

            papers = search_papers(
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                sort_opt=sort_option.lower(),
                category_option=category_option,
            )

            st.session_state["search_results"]["searched"] = True
            st.session_state["search_results"]["papers"] = papers

            ui_clusters, ui_top_words = {}, {}

            if len(papers) >= MIN_PAPERS_FOR_CLUSTERING:
                clusters, top_words, metrics = get_paper_clusters_hdbscan(papers)
                st.session_state["search_results"]["metrics"] = metrics
                # Optimal K is for K-Means
                st.session_state["search_results"]["optimal_k"] = get_optimal_k(papers)

                # Shift clusters by 1 for UI representation
                ui_clusters = {k + 1: v for k, v in clusters.items()}
                ui_top_words = {k + 1: v for k, v in top_words.items()}

            # Add All Papers as Cluster 0
            ui_clusters[0] = papers
            ui_top_words[0] = []
            st.session_state["search_results"]["clusters"] = ui_clusters
            st.session_state["search_results"]["top_words"] = ui_top_words
        except (InvalidKeywordError, InvalidDateRangeError, TooManyKeywordsError) as e:
            st.error(f"⚠️ {str(e)}")
            logging.error(f"Input error: {str(e)}")
        except ArxivFetchingError as e:
            st.warning("We couldn't fetch papers right now. Please try again in a few minutes.")
            logging.error(f"Fetching error: {str(e)}")
        except Exception as e:
            st.error("⚠️ An unexpected error occurred")
            logging.error(f"Unexpected error: {str(e)}")

if st.session_state["search_results"]["searched"] and st.session_state["search_results"]["papers"]:
    papers = st.session_state["search_results"]["papers"]
    clustering_active = len(papers) >= MIN_PAPERS_FOR_CLUSTERING
    
    col_results = st.columns(1)[0]
    with col_results:
        # Show number of results
        st.markdown(
            f"<div class='results-count'>{len(st.session_state['search_results']['papers'])} Papers Found</div>",
            unsafe_allow_html=True,
        )

    # Show each paper
    clusters = st.session_state["search_results"].get("clusters")
    top_words = st.session_state["search_results"].get("top_words")

    # Define this variable here to avoid error when "Show Clusters" is not active
    cluster_ids = sorted(list(clusters.keys()))

    if clustering_active:

        col_toggle = st.columns(1)[0]
        with col_toggle:
            show_clusters = st.toggle("Show Clusters", value=True)

        if show_clusters:
            col_radio, col_button = st.columns([0.55, 1.2], vertical_alignment="center")

            with col_radio:
                method = st.radio(
                    "Clustering Method",
                    ["Auto-detect clusters", "Specify number of clusters"],
                    # Set the current value to session_state["clustering_method"]
                    key="clustering_method",
                    label_visibility="visible",
                    help=("Controls how papers are grouped. Auto-detect automatically discovers "
                        "topic groups from the data. Specify number allows you to manually set "
                        "how many clusters will be created.")
                )
            with col_button:
                cluster_button = st.button("Cluster")

            if method == "Specify number of clusters":
                st.toast(f"Based on our analysis, the optimal number of clusters is {st.session_state['search_results']['optimal_k']}.", duration="short", icon="💡")

                col_slider, _ = st.columns([0.6, 0.7]) 
                with col_slider:
                    n_clusters = st.slider(
                        "Number of Clusters", 
                        min_value=1, 
                        max_value=MAX_NUMBER_OF_CLUSTERS, 
                        value=st.session_state['search_results']['optimal_k']
                    )

            if cluster_button:
                # Clusters are returned as a dictionary with keys from 0 to N-1
                # We shift them by 1 for UI representation

                if method == "Auto-detect clusters":
                    clusters, top_words, metrics = get_paper_clusters_hdbscan(st.session_state['search_results']['papers'])
                else:
                    clusters, top_words, metrics = get_papers_kmeans(st.session_state['search_results']['papers'], n_clusters)

                # Shift clusters by 1 for UI representation
                ui_clusters = {k + 1: v for k, v in clusters.items()}
                ui_top_words = {k + 1: v for k, v in top_words.items()}

                # Add All Papers as Cluster 0
                ui_clusters[0] = st.session_state['search_results']['papers']
                ui_top_words[0] = []

                st.session_state["search_results"]["clusters"] = ui_clusters
                st.session_state["search_results"]["top_words"] = ui_top_words
                st.session_state["search_results"]["metrics"] = metrics

                # Update local variables so the immediate render correctly uses 1..N instead of 0..N-1
                clusters = ui_clusters
                top_words = ui_top_words

            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

            cluster_ids = sorted(list(clusters.keys()))
            color_map = build_cluster_color_map(cluster_ids)

            # Filter out cluster 0 for the top words row since it's the 'All Papers' container
            display_cluster_ids = [cid for cid in cluster_ids if cid != 0]

            # Show top words above the cluster graph
            for row_start in range(0, len(display_cluster_ids), 3):
                row_ids = display_cluster_ids[row_start:row_start + 3]
                cols = st.columns(3)
                
                for col, cid in zip(cols, row_ids):
                    with col:
                        badges = ''.join([f"<span class='cluster-keyword-badge' style='font-size:0.65rem; padding:0.1rem 0.4rem;'>{w}</span>" for w in top_words.get(cid, [])])

                        # The cid will never be 0 here
                        if is_noise_cluster(cid):
                            cluster_label = HDBSCAN_NOISE_CLUSTER_NAME
                            badges = "<span class='cluster-keyword-badge' style='font-size:0.65rem; padding:0.1rem 0.4rem;'>Uncategorized Papers</span>"
                        else:
                            cluster_label = f"Cluster {cid}"
                        
                        cluster_color = color_map.get(cluster_label, "var(--uofsc-garnet)")

                        st.markdown(
                            f"<div style='display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.3rem;'>"
                            f"<span style='font-size:0.75rem; font-weight:700; color:{cluster_color};'>{cluster_label}</span>"
                            f"{badges}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            try:
                # Flatten the clusters to match the dataframe indices
                ordered_papers = []
                all_labels = []
                for label, p_list in clusters.items():
                    if label == 0:
                        continue
                    for paper in p_list:
                        ordered_papers.append(paper)
                        all_labels.append(str(label))

                metrics = st.session_state["search_results"].get("metrics")
                create_cluster_viz(ordered_papers, all_labels, {
                    "DBI": metrics["DBI"],
                    "CHI": metrics["CHI"],
                    "SIL": metrics["SIL"]
                    })
            except Exception as e:
                st.warning("Visualization failed")
                logging.error(f"Visualization failed: {e}")
    else:
        st.info("Not enough papers to visualize clusters.")

    # Show papers by cluster
    tab_titles = []
    for cid in cluster_ids:
        if cid == 0:
            tab_titles.append("All Papers")
        elif is_noise_cluster(cid):
            tab_titles.append(HDBSCAN_NOISE_CLUSTER_NAME)
        else:
            tab_titles.append(f"Cluster {cid}")
            
    tabs = st.tabs(tab_titles)

    for tab, cluster_id in zip(tabs, cluster_ids):
        with tab:
            if clustering_active:
                base_html = """
                <div class='cluster-info'>
                    <span class='cluster-paper-count'>📄 {paper_count} Papers</span>
                    {extra}
                </div>
                """
                paper_count = len(clusters[cluster_id])

                if cluster_id == 0:
                    # All papers cluster don't have keywords
                    st.markdown(
                        base_html.format(
                            paper_count=paper_count,
                            extra=""
                        ),
                        unsafe_allow_html=True
                    )
                elif is_noise_cluster(cluster_id):
                    st.markdown(
                        base_html.format(
                            paper_count=paper_count,
                            extra="<span class='cluster-uncategorized'>Papers not assigned to any of the clusters (Uncategorized)</span>"
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    keywords_badges = ''.join([f"<span class='cluster-keyword-badge'>{word}</span>" for word in top_words.get(cluster_id, [])])
                    st.markdown(
                        base_html.format(
                            paper_count=paper_count,
                            extra=f"<div class='cluster-keywords'><span class='cluster-keywords-label'>Key Terms:</span>{keywords_badges}</div>"
                        ),
                        unsafe_allow_html=True
                    )

            for paper in clusters[cluster_id]:
                paper_date = paper.published.strftime("%d/%m/%Y")
                other_cats_html = "".join(f'<div class="paper-other-categories">{cat}</div>' for cat in paper.categories[1:])

                paper_html = f"""
                <div class="paper-card">
                    <div class="paper-title">{paper.title}</div>
                    <div class="paper-categories">
                        <div class="paper-main-category tooltip">{paper.main_category}<span class="tooltiptext">Primary Category</span></div>{other_cats_html}
                    </div>
                    <div class="paper-metadata">
                        <span class="metadata-item">📅 {paper_date}</span>
                        <span class="metadata-item">🔗 <a href="{paper.link}">View on arXiv</a></span>
                    </div>
                </div>
                """
                st.markdown(paper_html, unsafe_allow_html=True)

                with st.expander("View details"):
                    authors_str = ", ".join(paper.authors)
                    st.markdown(
                        f"""
                        <div class='paper-authors'><strong>Authors:</strong> {authors_str}</div>
                        [📄 <a href='{paper.pdf_link}'>PDF</a>]
                        <div class='paper-abstract'><strong>Abstract</strong><br>{paper.abstract}</div>
                    """,
                        unsafe_allow_html=True,
                    )

# Empty state
elif st.session_state["search_results"]["searched"] and not st.session_state["search_results"]["papers"]:
    st.info("🔍 No papers found. Try adjusting your keywords or search period.")
