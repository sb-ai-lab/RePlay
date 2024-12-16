"""
Painless creation and retrieval of Spark sessions
"""

import logging
import os
import sys
from math import floor
from typing import Any, Dict, Optional

import psutil

from .types import PYSPARK_AVAILABLE, MissingImportType

if PYSPARK_AVAILABLE:
    from pyspark import __version__ as pyspark_version
    from pyspark.sql import SparkSession
else:
    SparkSession = MissingImportType


def get_spark_session(
    spark_memory: Optional[int] = None,
    shuffle_partitions: Optional[int] = None,
    core_count: Optional[int] = None,
) -> SparkSession:
    """
    Get default SparkSession

    :param spark_memory: GB of memory allocated for Spark;
        70% of RAM by default.
    :param shuffle_partitions: number of partitions for Spark; triple CPU count by default
    :param core_count: Count of cores to execute, ``-1`` means using all available cores.
        If ``None`` then checking out environment variable ``REPLAY_SPARK_CORE_COUNT``,
        if variable is not set then using ``-1``.
        Default: ``None``.
    """
    if os.environ.get("SCRIPT_ENV", None) == "cluster":  # pragma: no cover
        return SparkSession.builder.getOrCreate()

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    if os.environ.get("REPLAY_JAR_PATH"):  # pragma: no cover
        path_to_replay_jar = os.environ.get("REPLAY_JAR_PATH")
    else:
        if pyspark_version.startswith("3.1"):  # pragma: no cover
            path_to_replay_jar = (
                "https://repo1.maven.org/maven2/io/github/sb-ai-lab/replay_2.12/3.1.3/replay_2.12-3.1.3.jar"
            )
        elif pyspark_version.startswith(("3.2", "3.3")):  # pragma: no cover
            path_to_replay_jar = "https://repo1.maven.org/maven2/io/github/sb-ai-lab/replay_2.12/3.2.0_als_metrics/replay_2.12-3.2.0_als_metrics.jar"
        elif pyspark_version.startswith("3.4"):  # pragma: no cover
            path_to_replay_jar = "https://repo1.maven.org/maven2/io/github/sb-ai-lab/replay_after_fix_2.12/0.1/replay_after_fix_2.12-0.1.jar"
        else:  # pragma: no cover
            path_to_replay_jar = (
                "https://repo1.maven.org/maven2/io/github/sb-ai-lab/replay_2.12/3.1.3/replay_2.12-3.1.3.jar"
            )
            logging.warning(
                "Replay ALS model support only spark 3.1-3.4 versions! Replay will use "
                "'https://repo1.maven.org/maven2/io/github/sb-ai-lab/replay_2.12/3.1.3/replay_2.12-3.1.3.jar' "
                "in 'spark.jars' property."
            )

    if core_count is None:  # checking out env variable
        core_count = int(os.environ.get("REPLAY_SPARK_CORE_COUNT", "-1"))
    if spark_memory is None:
        env_var = os.environ.get("REPLAY_SPARK_MEMORY")
        spark_memory = int(env_var) if env_var is not None else floor(psutil.virtual_memory().total / 1024**3 * 0.7)
    if shuffle_partitions is None:
        shuffle_partitions = os.cpu_count() * 3
    driver_memory = f"{spark_memory}g"
    user_home = os.environ["HOME"]
    spark_session_builder = (
        SparkSession.builder.config("spark.driver.memory", driver_memory)
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config("spark.jars", path_to_replay_jar)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.local.dir", os.path.join(user_home, "tmp"))
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config("spark.kryoserializer.buffer.max", "256m")
        .config("spark.files.overwrite", "true")
        .master(f"local[{'*' if core_count == -1 else core_count}]")
    )

    return spark_session_builder.getOrCreate()


def logger_with_settings() -> logging.Logger:
    """Set up default logging"""
    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.WARN)
    logger = logging.getLogger("replay")
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


class Borg:
    """
    This class allows to share objects between instances.
    """

    _shared_state: Dict[str, Any] = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class State(Borg):
    """
    All modules look for Spark session via this class. You can put your own session here.
    """

    def __init__(
        self,
        session: Optional[SparkSession] = None,
    ):
        Borg.__init__(self)
        if not hasattr(self, "logger_set"):
            self.logger = logger_with_settings()
            self.logger_set = True

        if session is None:
            if not hasattr(self, "session"):
                self.session = get_spark_session()
        else:
            self.session = session
