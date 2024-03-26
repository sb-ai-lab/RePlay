import logging


def get_logger(
    name,
    level=logging.INFO,
    format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
    date_format="%Y-%m-%d %H:%M:%S",
    file=False,
):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler() if not file else logging.FileHandler(name)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
