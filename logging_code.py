import logging
import os

def setup_logging(name):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    # Save to file
    fh = logging.FileHandler(f"logs/{name}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Print to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger