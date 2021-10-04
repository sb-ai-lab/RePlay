"""
Painless creation and retrieval of Spark sessions
"""

import logging
import os
import sys
from math import floor
from typing import Any, Dict, Optional

import psutil
import torch
from pyspark.sql import SparkSession


def get_spark_session(
    spark_memory: Optional[int] = None,
    shuffle_partitions: Optional[int] = None,
) -> SparkSession:
    """
    Get default SparkSession

    :param spark_memory: GB of memory allocated for Spark;
        half of RAM by default.
    :param shuffle_partitions: number of partitions for Spark; number of CPU by default
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    if spark_memory is None:
        spark_memory = floor(psutil.virtual_memory().total / 1024 ** 3 * 0.7)
    if shuffle_partitions is None:
        shuffle_partitions = os.cpu_count() * 3
    driver_memory = f"{spark_memory}g"
    user_home = os.environ["HOME"]
    spark = (
        SparkSession.builder.config("spark.driver.memory", driver_memory)
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.local.dir", os.path.join(user_home, "tmp"))
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .master("local[*]")
        .enableHiveSupport()
        .getOrCreate()
    )
    return spark


def logger_with_settings() -> logging.Logger:
    """Set up default logging"""
    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.WARN)
    ignite_engine_logger = logging.getLogger("ignite.engine.engine.Engine")
    ignite_engine_logger.setLevel(logging.WARN)
    logger = logging.getLogger("replay")
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    return logger


# pylint: disable=too-few-public-methods
class Borg:
    """
    This class allows to share objects between instances.
    """

    _shared_state: Dict[str, Any] = {}

    def __init__(self):
        self.__dict__ = self._shared_state


# pylint: disable=too-few-public-methods
class State(Borg):
    """
    All modules look for Spark session via this class. You can put your own session here.

    Other parameters are stored here too: ``default device`` for ``pytorch`` (CPU/CUDA)
    """

    def __init__(
        self,
        session: Optional[SparkSession] = None,
        device: Optional[torch.device] = None,
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

        if device is None:
            if not hasattr(self, "device"):
                if torch.cuda.is_available():
                    self.device = torch.device(
                        f"cuda:{torch.cuda.current_device()}"
                    )
                else:
                    self.device = torch.device("cpu")
        else:
            self.device = device
