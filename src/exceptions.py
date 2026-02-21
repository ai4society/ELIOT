class ArxivSearcherError(Exception):
    """Base exception for all errors in the arxiv searcher."""
    pass

class InvalidCategoryError(ArxivSearcherError):
    """Raised when an invalid category is provided."""
    pass

class InvalidKeywordError(ArxivSearcherError):
    """Raised when keywords are empty or contain invalid characters."""
    pass

class InvalidDateRangeError(ArxivSearcherError):
    """Raised when the end date is before the start date."""
    pass

class TooManyKeywordsError(ArxivSearcherError):
    """Raised when the number of keywords exceeds the limit."""
    pass

class ArxivFetchingError(ArxivSearcherError):
    """Raised when there is an issue fetching results from the arXiv API."""
    pass
