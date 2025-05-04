from contextlib import contextmanager
import time
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


@contextmanager
def timer(task: str):
    """Simple timing decorator."""
    start = time.perf_counter()
    logging.info(f"[{task}] started …")
    yield
    elapsed = time.perf_counter() - start
    logging.info(f"[{task}] finished in {elapsed:.2f}s")
