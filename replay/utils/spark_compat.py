import re

from replay.utils.types import PYSPARK_AVAILABLE


def get_spark_major_version() -> int | None:
    """
    Return installed PySpark major version.

    :return: major version or ``None`` when PySpark is unavailable.
    """
    if not PYSPARK_AVAILABLE:
        return None

    from pyspark import __version__ as pyspark_version

    match = re.match(r"(\d+)", pyspark_version)
    if not match:
        return None
    return int(match.group(1))


def is_spark_4_or_higher() -> bool:
    """
    Check whether current runtime uses Spark 4+.
    """
    major = get_spark_major_version()
    return major is not None and major >= 4
