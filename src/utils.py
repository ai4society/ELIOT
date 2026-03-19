import time
import logging
from functools import wraps

def timeit(func):
    """Decorador para medir o tempo exato de execução de funções pesadas."""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        logging.info(f"The function '{func.__name__}' finished in {total_time:.4f} seconds.")
        return result
    return timeit_wrapper
