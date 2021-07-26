# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession

import replay.session_handler
from replay import utils
from tests.utils import spark


def test_func_get():
    vector = np.arange(2)
    assert utils.func_get(vector, 0) == 0.0


def test_get_spark_session():
    spark = replay.session_handler.get_spark_session(1)
    assert isinstance(spark, SparkSession)
    assert spark.conf.get("spark.driver.memory") == "1g"
    assert replay.session_handler.State(spark).session is spark
    assert replay.session_handler.State().session is spark


def test_convert():
    dataframe = pd.DataFrame(
        [[1, "a", 3.0], [3, "b", 5.0]], columns=["a", "b", "c"]
    )
    spark_df = utils.convert2spark(dataframe)
    pd.testing.assert_frame_equal(dataframe, spark_df.toPandas())
    assert utils.convert2spark(spark_df) is spark_df


def del_files_by_pattern(directory: str, pattern: str) -> None:
    """
    Удаляет файлы из директории в соответствии с заданным паттерном имени файла
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            os.remove(os.path.join(directory, filename))


def find_file_by_pattern(directory: str, pattern: str) -> Optional[str]:
    """
    Возвращает путь к первому найденному файлу в директории, соответствующему паттерну,
    или None, если таких файлов нет
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            return os.path.join(directory, filename)
    return None
