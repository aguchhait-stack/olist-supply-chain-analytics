import logging
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(levelname)s: %(message)s")