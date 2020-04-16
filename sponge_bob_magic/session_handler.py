"""
Этот модуль позволяет безболезненно создавать и получать спарк сессии.
"""

import logging
import os
from math import floor
from typing import Optional

import psutil
import torch
from pyspark.sql import SparkSession


def get_spark_session(spark_memory: Optional[int] = None) -> SparkSession:
    """
    инициализирует и возращает SparkSession с "годными" параметрами по
    умолчанию (для пользователей, которые не хотят сами настраивать Spark)

    :param spark_memory: количество гигабайт оперативной памяти, которую нужно выделить под Spark;
        если не задано, выделяется половина всей доступной памяти
    """
    if spark_memory is None:
        spark_memory = floor(psutil.virtual_memory().total / 1024 ** 3 / 2)
    if os.environ["PYTEST_RUNNING"] == "Y":
        driver_memory = "512m"
        shuffle_partitions = "1"
    else:
        driver_memory = f"{spark_memory}g"
        shuffle_partitions = "200"
    user_home = os.environ["HOME"]
    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
    spark = (
        SparkSession
        .builder
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.local.dir", os.path.join(user_home, "tmp"))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "localhost")
        .config("spark.sql.execution.arrow.enabled", "true")
        .master(f"local[*]")
        .enableHiveSupport()
        .getOrCreate()
    )
    return spark


def logger_settings():
    """Настройка логгеров и изменение их уровня"""
    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.WARN)
    ignite_engine_logger = logging.getLogger("ignite.engine.engine.Engine")
    ignite_engine_logger.setLevel(logging.WARN)
    sponge_logger = logging.getLogger("sponge_bob_magic")
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    sponge_logger.addHandler(hdlr)
    sponge_logger.setLevel(logging.DEBUG)


class Borg:
    """
    Обеспечивает доступ к расшаренному состоянию
    """
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class State(Borg):
    """
    В этот класс можно положить свою спарк сессию, чтобы она была доступна модулям библиотеки.
    Каждый модуль, которому нужна спарк сессия, будет искать её здесь и создаст дефолтную сессию,
    если ни одной не было создано до сих пор.

    Здесь же хранится default device для pytorch (CPU или CUDA, если доступна).
    """
    def __init__(
            self,
            session: Optional[SparkSession] = None,
            device: Optional[torch.device] = None
    ):
        Borg.__init__(self)
        if not hasattr(self, "logger_set"):
            logger_settings()
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
