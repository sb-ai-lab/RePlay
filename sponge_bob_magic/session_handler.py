"""
Этот модуль позволяет безболезненно создавать и получать спарк сессии.
"""

import logging
import os
from math import floor
from typing import Optional

import psutil
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
    spark_cores = "*"
    user_home = os.environ["HOME"]
    spark = (
        SparkSession
        .builder
        .config("spark.driver.memory", f"{spark_memory}g")
        .config("spark.local.dir", os.path.join(user_home, "tmp"))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .master(f"local[{spark_cores}]")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.WARN)
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    return spark


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
    """
    def __init__(self, session: Optional[SparkSession] = None):
        Borg.__init__(self)
        if session is None:
            if 'session' not in self.__dict__.keys():
                self.session = get_spark_session()
        else:
            self.session = session
