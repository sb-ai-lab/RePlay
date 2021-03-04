# pylint: disable-all
import os
import re

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


import replay.session_handler
from replay import utils


def test_func_get():
    vector = np.arange(2)
    assert utils.func_get(vector, 0) == 0.0


def test_get_spark_session():
    spark = replay.session_handler.get_spark_session(1)
    assert isinstance(spark, SparkSession)
    assert spark.conf.get("spark.driver.memory") == "1g"


def test_convert():
    df = pd.DataFrame([[1, "a", 3.0], [3, "b", 5.0]], columns=["a", "b", "c"])
    sf = utils.convert2spark(df)
    pd.testing.assert_frame_equal(df, sf.toPandas())
    assert utils.convert2spark(sf) is sf


def del_files_by_pattern(directory: str, pattern: str) -> None:
    """
    Удаляет файлы из директории в соответствии с заданным паттерном имени файла
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            os.remove(os.path.join(directory, filename))
