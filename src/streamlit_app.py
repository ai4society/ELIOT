import requests
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import List
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from arxiv_searcher import ARXIV_CATEGORIES, Paper, search

from exceptions import (
    ArxivFetchingError,
    ArxivSearcherError,
    InvalidDateRangeError,
    InvalidKeywordError,
    TooManyKeywordsError,
)
from ml.ml_utils import (
    MAX_NUMBER_OF_CLUSTERS,
    MIN_PAPERS_FOR_CLUSTERING,
    get_optimal_k,
    get_paper_clusters_hdbscan,
    get_papers_clusters_agglomerative,
)


class ClusteringMethod(str, Enum):
    AUTO_DETECT = "Auto-detect clusters"
    SPECIFY_NUMBER = "Specify number of clusters"


@st.cache_data(ttl=timedelta(hours=2), max_entries=50, show_spinner=False)
def search_papers(
    keywords: str, start_date: date, end_date: date, sort_opt: str, category_option: str, max_results: int
) -> List[Paper]:
    return search(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_opt,
        category=category_option,
        max_results=max_results,
    )


def is_noise_cluster(cid: int) -> bool:
    """Returns True if cid represents the HDBSCAN noise/uncategorized cluster.
    After the +1 shift applied in streamlit_app, the noise sentinel (-1) becomes 0.
    Agglomerative clustering clusters are strictly 1..N after the shift, so 0 is uniquely noise.
    """
    return cid == 0


def get_cluster_name(cid: int) -> str:
    if cid == ALL_PAPERS_TAB_KEY:
        return "All Papers"
    return "Uncategorized" if is_noise_cluster(cid) else f"Cluster {cid}"


def sort_cluster_ids(cluster_ids: List[int]) -> List[int]:
    all_papers = [ALL_PAPERS_TAB_KEY] if ALL_PAPERS_TAB_KEY in cluster_ids else []
    normal_clusters = sorted([c for c in cluster_ids if c > 0])
    noise_cluster = [c for c in cluster_ids if is_noise_cluster(c)]
    return all_papers + normal_clusters + noise_cluster

def create_cluster_viz(ordered_papers: List[Paper], all_labels: List[str], metrics: dict):
    years = [p.published.year for p in ordered_papers]
    cluster_nums = [int(label) for label in all_labels]
    K = len(set(all_labels))

    np.random.seed(42)
    # Add jitter to avoid overlapping points since both axes are discrete
    jitter_x = np.random.uniform(-0.3, 0.3, size=len(ordered_papers))
    jitter_y = np.random.uniform(-0.3, 0.3, size=len(ordered_papers))

    df_viz = pd.DataFrame({
        "Year": years,
        "Cluster": [get_cluster_name(int(label)) for label in all_labels],
        "x": np.array(years) + jitter_x,
        "y": np.array(cluster_nums) + jitter_y,
        "Title": [p.title for p in ordered_papers],
        "Date": [p.published.strftime("%Y-%m-%d") for p in ordered_papers]
    })

    df_viz["Papers this year"] = df_viz.groupby(["Year", "Cluster"])["Title"].transform("count")

    fig = px.scatter(
        df_viz,
        x="x", y="y",
        color="Cluster",
        color_discrete_map=color_map,
        hover_data={
            "Title": True,
            "Date": True,
            "Cluster": True,
            "Papers this year": True,
            "x": False,
            "y": False
        },
        title="Evolution of Research Topics Over Time",
    )

    fig.update_traces(
        marker=dict(
            size=11,
            opacity=0.90,
            line=dict(width=0.8, color="white")
        )
    )

    fig.update_layout(
        font=dict(family="Inter, Arial, sans-serif", size=13),
        title=dict(font=dict(size=16), x=0.01),
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            title_text="Cluster",
        ),
        xaxis=dict(
            title="Year",
            tickmode="linear",
            dtick=1,
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Cluster ID",
            tickmode="array",
            tickvals=sorted(list(set(cluster_nums))),
            ticktext=[get_cluster_name(c) for c in sorted(list(set(cluster_nums)))],
            showgrid=True,
            zeroline=False,
        ),
        hovermode="closest",
        margin=dict(r=200),
    )

    fig.add_annotation(
        x=1.01,
        y=0,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font=dict(size=11),
        text=(
            "Metrics (Based on All Clusters):<br>"
            f"     <a href='https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index' target='_blank'><b>Davies-Bouldin</b></a>: {metrics['DBI']:.2f}<br>"
            f"     <a href='https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index' target='_blank'><b>Calinski-Harabasz</b></a>: {metrics['CHI']:.2f}<br>"
            f"     <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'><b>Silhouette Score</b></a>: {metrics['SIL']:.3f}"
        ),
    )

    base_height = 300
    height_per_cluster = 40
    height = base_height + K * height_per_cluster
    height = max(500, min(height, 1000))

    return st.plotly_chart(fig, height=height, config={"displayModeBar": True})


def build_cluster_color_map(cluster_ids: List[int]):
    palette = px.colors.qualitative.Plotly
    color_map = {}
    for i, cid in enumerate(cluster_ids):
        color_map[get_cluster_name(cid)] = palette[i % len(palette)]
    return color_map


def get_default_session_state():
    return {
        "papers": [], 
        "searched": False, 
        "clusters": {}, 
        "top_words": {}, 
        "metrics": {}, 
        "optimal_k": 1,
        "show_optimal_k": True,
        "show_all_clusters": False,
    }


GOATCOUNTER_DOMAIN = "bernardo.goatcounter.com"
GOATCOUNTER_PATH = "/eliot"
# Manual offset based on the visit count shown in Streamlit Cloud analytics
# Before GoatCounter was added
INITIAL_VISIT_COUNT = 58

def register_goatcounter_visit() -> None:
    components.html(
        f"""
        <script>
            window.goatcounter = {{
                allow_frame: true,
                path: "{GOATCOUNTER_PATH}",
                title: "Eliot"
            }};
        </script>

        <script data-goatcounter="https://{GOATCOUNTER_DOMAIN}/count"
                async src="https://gc.zgo.at/count.js"></script>
        """,
        height=1,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_goatcounter_visitors() -> str:
    """Fetch the app visit count from GoatCounter, cached for 30 minutes."""

    url = f"https://{GOATCOUNTER_DOMAIN}/counter/{GOATCOUNTER_PATH.lstrip('/')}.json"

    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        return str(int(data["count"]) + INITIAL_VISIT_COUNT)
    except Exception as e:
        print(f"Error getting GoatCounter views: {e}")
        return "-"


st.set_page_config(
    page_title="Eliot", page_icon="📚", layout="centered"
)

css_path = Path(__file__).parent / "styles.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

register_goatcounter_visit()

st.markdown(
    '<div class="header-bar">'
    '<a href="https://ai4society.github.io/" target="_blank" class="header-link">AI4Society</a>'
    f'<span class="views-badge">App Views: {fetch_goatcounter_visitors()}</span>'
    '</div>',
    unsafe_allow_html=True,
)

ORDER_BY_OPTIONS = {
    "Relevance": "relevance",
    "Submitted Date": "submitted",
}

DEFAULT_KEYWORDS = "large language models, multi-agent systems"

# Special key for the "All Papers" tab.
# This is NOT related to HDBSCAN's noise label (-1 pre-shift / 0 post-shift).
ALL_PAPERS_TAB_KEY = -1

@st.cache_data(ttl=timedelta(hours=24), show_spinner=False)
def load_default_results() -> dict:
    """
    Loads the default data the first time the page loads.
    This result is cached for 24 hours to serve as the initial data 
    and as a fallback for ArXiv API rate limits.
    """
    sd = date(date.today().year - 1, date.today().month, date.today().day)
    ed = date.today()
    cat = "Computer Science"
    
    papers = search(
        keywords=DEFAULT_KEYWORDS,
        start_date=sd, 
        end_date=ed, 
        sort_by=ORDER_BY_OPTIONS["Relevance"], 
        category=cat,
        max_results=300
    )
    
    res = get_default_session_state()
    res["searched"] = True
    res["papers"] = papers
    
    clusters, top_words, metrics = get_paper_clusters_hdbscan(papers)
    res["metrics"] = metrics
    res["optimal_k"] = get_optimal_k(papers)
    res["applied_clustering_method"] = ClusteringMethod.AUTO_DETECT.value
    
    ui_clusters = {k + 1: v for k, v in clusters.items()}
    ui_top_words = {k + 1: v for k, v in top_words.items()}
 
    ui_clusters[ALL_PAPERS_TAB_KEY] = papers
    ui_top_words[ALL_PAPERS_TAB_KEY] = []
    
    res["clusters"] = ui_clusters
    res["top_words"] = ui_top_words
    return res

# Initialize session state (on startup we use the cached default query)
if "search_results" not in st.session_state:
    try:
        st.session_state["search_results"] = load_default_results()
        st.session_state["clustering_method"] = ClusteringMethod.AUTO_DETECT.value
    except Exception as e:
        logging.error(f"Failed to load initial cache: {e}")
        st.session_state["search_results"] = get_default_session_state()

st.title("Eliot", anchor=False)

st.markdown(
    "<div class='page-subtitle'>Interactively Exploring Fast-Changing Scientific Literature Trends with Online Data and ML</div>",
    unsafe_allow_html=True,
)

col_keywords, col_category = st.columns([2.5, 1])

with col_keywords:
    keywords = st.text_input(
        "Search Keywords or Phrases",
        placeholder=f"{DEFAULT_KEYWORDS}",
        help=(
            "Enter comma-separated keywords, each may contain multiple words. "
            "Results include only papers containing all keywords. "
            f"If empty, defaults to: {DEFAULT_KEYWORDS}."
        ),
    )

with col_category:
    category_option = st.selectbox(
        "Category",
        ARXIV_CATEGORIES.keys(),
        index=0,
        help="Focus your search on a specific research field",
    )

col_start_date, col_end_date, col_order_by, col_max_results, col_search_bt = st.columns(
    [1, 1, 1.5, 1.5, 1], vertical_alignment="bottom"
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

with col_max_results:
    max_results = st.slider("Max Papers to Retrieve", min_value=20, max_value=500, value=300, step=10)

with col_search_bt:
    search_button = st.button("Search", use_container_width=True)

# On click
if search_button:
    previous_results = st.session_state.get("search_results", get_default_session_state())
    st.session_state["clustering_method"] = ClusteringMethod.AUTO_DETECT.value
    st.session_state["search_results"] = get_default_session_state()

    with st.spinner("🔎 Searching papers..."):
        try:
            if not keywords:
                keywords = DEFAULT_KEYWORDS

            papers = search_papers(
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                sort_opt=ORDER_BY_OPTIONS[sort_option],
                category_option=category_option,
                max_results=max_results,
            )
            logging.info(f"Search successful: found {len(papers)} papers for keywords '{keywords}' in category '{category_option}'")

            st.session_state["search_results"]["searched"] = True
            st.session_state["search_results"]["papers"] = papers

            ui_clusters, ui_top_words = {}, {}

            if len(papers) >= MIN_PAPERS_FOR_CLUSTERING:
                clusters, top_words, metrics = get_paper_clusters_hdbscan(papers)
                st.session_state["search_results"]["metrics"] = metrics
                st.session_state["search_results"]["optimal_k"] = get_optimal_k(papers)
                st.session_state["search_results"]["applied_clustering_method"] = ClusteringMethod.AUTO_DETECT.value

                # Shift real clusters (0..N-1) to (1..N).
                # Noise sentinel (-1) becomes 0 after +1 shift.
                ui_clusters = {k + 1: v for k, v in clusters.items()}
                ui_top_words = {k + 1: v for k, v in top_words.items()}

            # Add the "All Papers" tab (shows every paper, no cluster logic)
            ui_clusters[ALL_PAPERS_TAB_KEY] = papers
            ui_top_words[ALL_PAPERS_TAB_KEY] = []

            st.session_state["search_results"]["clusters"] = ui_clusters
            st.session_state["search_results"]["top_words"] = ui_top_words
        except (InvalidKeywordError, InvalidDateRangeError, TooManyKeywordsError) as e:
            st.error(f"⚠️ {str(e)}")
            logging.error(f"Input error: {str(e)}")

        except ArxivFetchingError as e:
            # When not able to fetch papers for the user's query, show the previous result
            if previous_results.get("searched"):
                st.warning("We couldn't fetch papers right now due to arXiv rate limiting. Please try again in a few minutes. In the meantime, we are safely keeping your currently displayed results.")
                st.session_state["search_results"] = previous_results
            else:
                st.warning("We couldn't fetch papers right now due to arXiv rate limiting. Please try again in a few minutes. Showing sample papers in the meantime.")
                st.session_state["search_results"] = load_default_results()
            logging.error(f"ArXiv Fetching Error details: {str(e)}")

        except Exception as e:
            st.error("⚠️ An unexpected error occurred")
            logging.error(
                f"\n ERROR DETAILS: {str(e)}\n\n"
                f"  User search parameters:\n"
                f"  keywords={keywords or DEFAULT_KEYWORDS}\n"
                f"  category={category_option}\n"
                f"  sort={sort_option}\n"
                f"  date_range={start_date} to {end_date}\n",
                exc_info=True,
            )

if st.session_state["search_results"].get("searched") and st.session_state["search_results"].get("papers"):
    papers = st.session_state["search_results"].get("papers")
    clustering_active = len(papers) >= MIN_PAPERS_FOR_CLUSTERING
    
    # Show number of results
    st.markdown(
        f"<div class='results-count'>📄 {len(st.session_state['search_results'].get('papers'))} Papers Found</div>",
        unsafe_allow_html=True,
    )

    # Show each paper
    clusters = st.session_state["search_results"].get("clusters")
    top_words = st.session_state["search_results"].get("top_words")

    # Define this variable here to avoid error when "Show Clusters" is not active
    cluster_ids = sort_cluster_ids(list(clusters.keys()))

    if clustering_active:
        col_show_clusters, col_method = st.columns([0.5, 0.5])
        with col_show_clusters:
            show_clusters = st.toggle("Cluster Settings & Analysis", value=True)

        if show_clusters:
            if st.session_state["search_results"].get("show_optimal_k"):
                st.toast(f"Based on our analysis, the optimal number of clusters is {st.session_state["search_results"].get("optimal_k")}.", duration="long", icon="💡")
                st.session_state["search_results"]["show_optimal_k"] = False

            with col_method:
                _, col_radio, _ = st.columns([1, 2, 1])
                with col_radio:
                    method = st.radio(
                        "Clustering Method",
                        [m.value for m in ClusteringMethod],
                        # Set the current value to session_state["clustering_method"]
                        key="clustering_method",
                        label_visibility="visible",
                        help=("Controls how papers are grouped. Auto-detect automatically discovers "
                            "topic groups from the data. Specify number allows you to manually set "
                            "how many clusters will be created.")
                    )

                if method == ClusteringMethod.SPECIFY_NUMBER.value:
                    n_clusters = st.slider(
                        "Number of Clusters", 
                        min_value=1, 
                        max_value=MAX_NUMBER_OF_CLUSTERS, 
                        value=st.session_state['search_results']['optimal_k']
                    )

                _, col_center, _ = st.columns([1, 2, 1])
                with col_center:
                    cluster_button = st.button("Run Clustering", use_container_width=True)
                st.markdown(f"<div style='margin-bottom:-1rem;'></div>", unsafe_allow_html=True)

            if cluster_button:
                # Clusters are returned as a dictionary with keys from 0 to N-1
                # We shift them by 1 for UI representation

                if method == ClusteringMethod.AUTO_DETECT.value:
                    clusters, top_words, metrics = get_paper_clusters_hdbscan(st.session_state['search_results']['papers'])
                else:
                    clusters, top_words, metrics = get_papers_clusters_agglomerative(st.session_state['search_results']['papers'], n_clusters)

                # Shift clusters numbers (0..N-1) to (1..N).
                # Noise cluster (-1) becomes 0 after +1 shift.
                ui_clusters = {k + 1: v for k, v in clusters.items()}
                ui_top_words = {k + 1: v for k, v in top_words.items()}

                # Add the "All Papers" tab (shows every paper, no cluster logic)
                ui_clusters[ALL_PAPERS_TAB_KEY] = st.session_state['search_results']['papers']
                ui_top_words[ALL_PAPERS_TAB_KEY] = []

                st.session_state["search_results"]["clusters"] = ui_clusters
                st.session_state["search_results"]["top_words"] = ui_top_words
                st.session_state["search_results"]["metrics"] = metrics
                st.session_state["search_results"]["applied_clustering_method"] = method

                # Update local variables so the immediate render correctly uses 1..N instead of 0..N-1
                clusters = ui_clusters
                top_words = ui_top_words

            cluster_ids = sort_cluster_ids(list(clusters.keys()))
            color_map = build_cluster_color_map(cluster_ids)

            # Filter out "All Papers" tab for the top words row
            display_cluster_ids = [cid for cid in cluster_ids if cid != ALL_PAPERS_TAB_KEY]

            # By default, to avoid visual clutter, only the top `optimal_k` clusters and their keywords are shown.
            # If the user clicks "Show All", this limitation is removed and all detected clusters will be rendered.
            optimal_k = st.session_state["search_results"].get("optimal_k")
            show_all = st.session_state["search_results"].get("show_all_clusters")
            applied_method = st.session_state["search_results"].get("applied_clustering_method")

            if applied_method == ClusteringMethod.SPECIFY_NUMBER.value:
                # When user specifies the number of clusters, show all of them
                visible_cluster_ids = display_cluster_ids
                total_clusters = len(display_cluster_ids)
            else:
                if not show_all:
                    visible_cluster_ids = display_cluster_ids[:optimal_k]
                else:
                    visible_cluster_ids = display_cluster_ids

                total_clusters = len(display_cluster_ids)

                # Show All / Show less toggle
                if total_clusters > optimal_k:
                    toggle_label = f"Show All" if not show_all else "Show Less"
                    st.markdown(
                        f"<div class='cluster-showing-label'>Showing {len(visible_cluster_ids)} of {total_clusters} clusters</div>",
                        unsafe_allow_html=True
                    )

                    if st.button(toggle_label, key="toggle_clusters"):
                        st.session_state["search_results"]["show_all_clusters"] = not show_all
                        st.rerun()

            # Show top words above the cluster graph
            for row_start in range(0, len(visible_cluster_ids), 3):
                row_ids = visible_cluster_ids[row_start:row_start + 3]
                cols = st.columns(3)
                
                for col, cid in zip(cols, row_ids):
                    with col:
                        badges = ''.join([f"<span class='cluster-keyword-badge' style='font-size:0.65rem; padding:0.1rem 0.4rem;'>{w}</span>" for w in top_words.get(cid, [])])
                        cluster_label = get_cluster_name(cid)
                        if is_noise_cluster(cid):
                            badges = "<span class='cluster-keyword-badge' style='font-size:0.65rem; padding:0.1rem 0.4rem;'>Uncategorized Papers</span>"
                        
                        cluster_color = color_map.get(cluster_label)

                        st.markdown(
                            f"<div style='display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap; margin-bottom:1.3rem;'>"
                            f"<span style='font-size:0.75rem; font-weight:700; color:{cluster_color};'>{cluster_label}</span>"
                            f"{badges}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
 
            try:
                # Flatten the clusters to match the dataframe indices (skip All Papers and noise)
                # Only include clusters within the visible set (respects optimal_k / show_all)
                ordered_papers = []
                all_labels = []
                for label, p_list in clusters.items():
                    if label in (ALL_PAPERS_TAB_KEY, 0):  # All Papers or Uncategorized
                        continue
                    if label not in visible_cluster_ids:  # Respect optimal_k gate
                        continue
                    for paper in p_list:
                        ordered_papers.append(paper)
                        all_labels.append(str(label))

                metrics = st.session_state["search_results"].get("metrics")
                create_cluster_viz(ordered_papers, all_labels, {
                    "DBI": metrics.get("DBI", 0.0),
                    "CHI": metrics.get("CHI", 0.0),
                    "SIL": metrics.get("SIL", 0.0),
                })
            except Exception as e:
                st.warning("Visualization failed")
                logging.error(f"Visualization failed: {e}")
    else:
        st.info("Not enough papers to visualize clusters.")

    # Show papers by cluster
    # When show_all is False, tabs only show optimal_k clusters + All Papers tab.
    # The noise cluster (0) is always included if present.
    if clustering_active and show_clusters:
        noise_ids = [c for c in cluster_ids if is_noise_cluster(c)]
        all_papers_ids = [c for c in cluster_ids if c == ALL_PAPERS_TAB_KEY]
        if show_all:
            visible_tab_ids = cluster_ids
        else:
            visible_tab_ids = all_papers_ids + sort_cluster_ids(visible_cluster_ids) + noise_ids
            # de-duplicate while preserving order
            seen = set()
            deduped = []
            for x in visible_tab_ids:
                if x not in seen:
                    seen.add(x)
                    deduped.append(x)
            visible_tab_ids = deduped
    else:
        visible_tab_ids = cluster_ids

    tab_titles = [get_cluster_name(cid) for cid in visible_tab_ids]

    if not tab_titles:
        logging.error(
            f"Empty tab_titles detected:\n"
            f"  keywords={keywords or DEFAULT_KEYWORDS}\n"
            f"  category={category_option}\n"
            f"  sort={sort_option}\n"
            f"  date_range={start_date} to {end_date}\n"
            f"  papers_count={len(papers)}\n"
            f"  clustering_active={clustering_active}\n"
            f"  visible_tab_ids={visible_tab_ids}\n"
            f"  cluster_ids={cluster_ids}\n"
            f"  session_searched={st.session_state['search_results'].get('searched')}\n"
            f"  session_metrics={st.session_state['search_results'].get('metrics')}"
        )
        # Fallback to ensure at least the "All Papers" tab is shown
        clusters[ALL_PAPERS_TAB_KEY] = papers
        visible_tab_ids = [ALL_PAPERS_TAB_KEY]
        tab_titles = [get_cluster_name(ALL_PAPERS_TAB_KEY)]
        st.warning("⚠️ Unable to display clusters. Please try searching again.")
    tabs = st.tabs(tab_titles)

    for tab, cluster_id in zip(tabs, visible_tab_ids):
        with tab:
            if clustering_active:
                base_html = """
                <div class='cluster-info'>
                    <span class='cluster-paper-count'>📄 {paper_count} Papers</span>
                    {extra}
                </div>
                """
                paper_count = len(clusters[cluster_id])

                if cluster_id == ALL_PAPERS_TAB_KEY:
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

            with st.container(height=800, border=True):
                sorted_papers = sorted(clusters[cluster_id], key=lambda p: p.published, reverse=True)
                for paper in sorted_papers:
                    paper_date = paper.published.strftime("%d/%m/%Y")
                    other_cats_html = "".join(f'<div class="paper-other-categories">{cat}</div>' for cat in paper.categories[1:])

                    paper_html = f"""
                    <div class="paper-card">
                        <div class="paper-title">{paper.title}</div>
                        <div class="paper-categories">
                            <div class="paper-main-category tooltip">{paper.main_category}<span class="tooltiptext">Primary Category</span></div>{other_cats_html}
                        </div>
                        <div class="paper-metadata">
                            <span>📅 {paper_date}</span>
                            <span>🔗 <a href="{paper.link}">View on arXiv</a></span>
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

            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #777">
                    Thank you to arXiv for use of its open access interoperability.
                </div>
                """,
                unsafe_allow_html=True,
            )

# Empty state
elif st.session_state["search_results"].get("searched") and not st.session_state["search_results"].get("papers"):
    st.info("🔍 No papers found. Try adjusting your keywords or search period.")
