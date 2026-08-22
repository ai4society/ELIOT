import time
import logging
from functools import wraps

def timeit(func):
    """Decorator to measure the exact execution time of functions."""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        logging.info(f"[TIMER] Finished executing '{func.__name__}'. Total time elapsed: {total_time:.4f} seconds.")
        return result
    return timeit_wrapper
