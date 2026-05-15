from dataclasses import dataclass
import json
import logging
import os
from datetime import datetime
from typing import List

import arxiv
from arxivql.taxonomy import categories_by_id
from arxivql import Query, Taxonomy as T

import pandas as pd
from exceptions import (
    InvalidCategoryError,
    InvalidKeywordError,
    InvalidDateRangeError,
    TooManyKeywordsError,
    ArxivFetchingError
)

# Load configuration
def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


config = load_config()

# Use environment variable if set, otherwise use config file
BASE_DIR: str = os.environ.get(
    "ARXIV_SEARCHER_BASE_DIR", os.path.dirname(os.path.abspath(__file__))
)

# Setup paths
LOG_DIR = os.path.join(BASE_DIR, config["log_dir"])

# Setup logging
_log_format = "%(asctime)s - %(levelname)s - %(message)s"
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

# File handler, local dev
os.makedirs(LOG_DIR, exist_ok=True)
_file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, f"arxiv_searcher_{datetime.now().strftime('%Y%m%d')}.log")
)
_file_handler.setFormatter(logging.Formatter(_log_format))
_logger.addHandler(_file_handler)

# Stream handler, visible in Streamlit Cloud logs
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(_log_format))
_logger.addHandler(_stream_handler)

MAXIMUM_KEYWORDS_ALLOWED = 12

# All categories available in arxiv
# Link https://arxiv.org/category_taxonomy
ARXIV_CATEGORIES = {
    "Computer Science": T.cs,
    "Mathematics": T.math,
    "Physics": [
        T.astro_ph, T.cond_mat, T.gr_qc, T.hep_ex, T.hep_lat, 
        T.hep_ph, T.hep_th, T.math_ph, T.nlin, T.nucl_ex, 
        T.nucl_th, T.physics, T.quant_ph,
    ],
    "Statistics": T.stat,
    "Electrical Eng. & Systems Sci.": T.eess,
    "Quantitative Biology": T.q_bio,
    "Quantitative Finance": T.q_fin,
    "Economics": T.econ
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str] 
    abstract: str
    published: datetime
    updated: datetime
    link: str
    pdf_link: str
    main_category: str
    categories: List[str]


def build_arxiv_query(keywords: List[str], category: str = "cs") -> Query:
    """
    Build an arXiv search query using arxivql api.

    Args:
        keywords (List[str]): List of user-provided keywords or phrases.
        category (str): High-level arXiv category key mapped to multiple subcategories.
    """
    
    if category not in ARXIV_CATEGORIES.keys():
        raise InvalidCategoryError(f"Invalid arxiv category. Categories available: {ARXIV_CATEGORIES.keys()}") 

    keyword_query = None
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue

        try:
            # 'all' prefix for each keyword. Although broader in theory, in practice results are dominated
            # by title/abstract matches, and the simpler query structure yields better recall
            clause = Query.all(kw)
        except ValueError:
            logging.warning(f"Skipping malformed keyword: '{kw}'")
            raise InvalidKeywordError("Keyword with double quotes or parentheses are not allowed.")
        
        if keyword_query is None:
            keyword_query = clause
        else:
            keyword_query &= clause
    
    # Categories: search in any of the mapped subcategories (OR)
    subcategories = ARXIV_CATEGORIES[category]
    category_query = Query.category(subcategories)
    return category_query & keyword_query


def search(keywords: str, start_date: datetime.date, end_date: datetime.date, sort_by: str, category: str, max_results: int = 300) -> List[Paper]:
    """
    Search arXiv for papers matching the specified criteria.

    Args:
        keywords (str): str of keywords for filtering (e.g "time-series, forecasting").
        start_date (datetime.date): Start date.
        end_date (datetime.date): End date.
        sort_by (str): Sorting method, either 'relevance' or 'submitted'.
        category (str): Specific research category
        max_results (int): Maximum number of papers to retrieve.
    """
    logging.info("Querying arXiv for papers.")

    if end_date < start_date:
        raise InvalidDateRangeError("End Date must be greater than or equal to Start Date")

    if not keywords:
        raise InvalidKeywordError("At least one keyword must be provided")

    keywords = [item.strip() for item in keywords.split(",")]
    if len(keywords) > MAXIMUM_KEYWORDS_ALLOWED:
        raise TooManyKeywordsError(f"Too many keywords provided ({len(keywords)}). Maximum allowed is {MAXIMUM_KEYWORDS_ALLOWED}.")

    client = arxiv.Client(
        page_size=300,
        delay_seconds=8,
        num_retries=4
    )
    query = build_arxiv_query(keywords=keywords, category=category)

    # Add date filtering
    query &= Query.submitted_date(start_date, end_date)

    logging.info(f"Keywords for filtering: {keywords}")
    logging.info(f"Date Range: {start_date} - {end_date}")
    logging.info(f"Query being used: {query}\n")

    arxiv_search = arxiv.Search(
        query=str(query), 
        max_results=max_results, 
        sort_by=arxiv.SortCriterion.Relevance if sort_by == "relevance" else arxiv.SortCriterion.SubmittedDate
    )

    try:
        papers = [
            Paper(
                arxiv_id=result.get_short_id(),
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published,
                updated=result.updated,
                link=result.entry_id,
                pdf_link=result.pdf_url,
                main_category=categories_by_id[result.primary_category].name,
                # Uses arxivql api to get the category name
                categories=[categories_by_id[cat].name for cat in result.categories]
            )
            for result in client.results(arxiv_search)
        ]
        return papers
    except Exception as e:
        error_str = str(e)
        if "Too Many Requests" in error_str or "429" in error_str:
            logging.warning("ArXiv returned 429 (rate limited).")
            raise ArxivFetchingError("Error fetching papers: Too Many Requests")
            
        logging.error("ArXiv Fetching Error details: %s", e)
        raise ArxivFetchingError("Error fetching papers")
