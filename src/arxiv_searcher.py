import argparse
from dataclasses import dataclass
import json
import logging
import os
from datetime import date, datetime
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
logging.basicConfig(
    filename=os.path.join(
        LOG_DIR, f"arxiv_searcher_{datetime.now().strftime('%Y%m%d')}.log"
    ),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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

    # Each keyword must be in title OR abstract
    keyword_query = None
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        # (ti:kw OR abs:kw)
        try:
            clause = Query.title(kw) | Query.abstract(kw)
        except ValueError:
            raise InvalidKeywordError("Keyword with double quotes or parentheses are not allowed.")
        
        if keyword_query is None:
            keyword_query = clause
        else:
            keyword_query &= clause
    
    # Categories: search in any of the mapped subcategories (OR)
    subcategories = ARXIV_CATEGORIES[category]
    category_query = Query.category(subcategories)
    return category_query & keyword_query


def search(keywords: str, start_date: datetime.date, end_date: datetime.date, sort_by: str, category: str) -> List[Paper]:
    """
    Search arXiv for papers matching the specified criteria.

    Args:
        keywords (str): str of keywords for filtering (e.g "time-series, forecasting").
        start_date (datetime.date): Start date.
        end_date (datetime.date): End date.
        sort_by (str): Sorting method, either 'relevance' or 'submitted'.
        category (str): Specific research category
    """
    logging.info("Querying arXiv for papers.")

    if end_date < start_date:
        raise InvalidDateRangeError("End Date must be greater than or equal to Start Date")

    if not keywords:
        raise InvalidKeywordError("At least one keyword must be provided")

    keywords = [item.strip() for item in keywords.split(",")]
    if len(keywords) > MAXIMUM_KEYWORDS_ALLOWED:
        raise TooManyKeywordsError(f"Too many keywords provided ({len(keywords)}). Maximum allowed is {MAXIMUM_KEYWORDS_ALLOWED}.")

    client = arxiv.Client()
    query = build_arxiv_query(keywords=keywords, category=category)

    # Add date filtering
    query &= Query.submitted_date(start_date, end_date)

    logging.info(f"Keywords for filtering: {keywords}")
    logging.info(f"Date Range: {start_date} - {end_date}")
    logging.info(f"Query being used: {query}\n")
    print(f"Query being used: {query}\n")

    search = arxiv.Search(
        query=str(query), 
        max_results=300, 
        sort_by=arxiv.SortCriterion.Relevance if sort_by == "relevance" else arxiv.SortCriterion.SubmittedDate
    )

    #'arxiv_id', 'authors', 'updated', 'pdf_link', 'main_category', and 'categories'
    # temp = [
    #     [
    #         result.get_short_id(),
    #         [a.name for a in result.authors],
    #         result.updated,
    #         result.pdf_url,
    #         categories_by_id[result.primary_category].name,
    #         [categories_by_id[cat].name for cat in result.categories],
    #         result.title,
    #         result.summary,
    #         result.published,
    #         result.entry_id
    #     ]
    #     for result in client.results(search)
    # ]

    # df = pd.DataFrame(temp, columns=["arxiv_id", "authors", "updated", "pdf_link", "main_category", "categories", "title", "abstract", "published", "link"])
    # ano_query = str(query).replace(" ", "_")
    # query_keywords = "_".join([kw.replace(" ", "_") for kw in keywords])
    # query_date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    # df.to_csv(f"papers_{query_date_range}_{query_keywords}.csv", index=False)

    try:
        return [
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
            for result in client.results(search)
        ]
    except Exception as e:
       logging.error("Error getting results: %s", e)
       raise ArxivFetchingError("Error fetching papers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local arXiv search extraction.")

    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="Comma-separated keywords for filtering (e.g., 'planning,PDDL').",
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=ARXIV_CATEGORIES.keys(),
        help="ArXiv categories",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--sort_by",
        type=str,
        default="relevance",
        choices=["relevance", "submitted"],
        help="Sorting method for arXiv results.",
    )

    args = parser.parse_args()

    end_date = args.end_date if args.end_date is not None else str(date.today())

    results = search(
        keywords=args.keywords,
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
        sort_by=args.sort_by,
        category=args.category
    )

    for i in range(len(results)):
        if i >= 5:
            break
        paper = results[i]
        print(f"[{i+1}] {paper.arxiv_id} - {paper.title}")
        print(f"Authors: {', '.join(paper.authors)}")
        print(paper.abstract)
        print()

    print()
    print(f"Total papers found: {len(results)}")