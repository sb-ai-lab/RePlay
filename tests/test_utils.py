# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
from datetime import datetime
from datetime import timezone
from functools import partial

import numpy as np
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType
import pytest

import replay.session_handler
from replay import utils
from tests.utils import spark, sparkDataFrameEqual, long_log_with_features

datetime = partial(datetime, tzinfo=timezone.utc)

different_timestamp_formats_data = [
    (
        [
            [1.0, 1],
            [300003.0, 300003],
            [0.0, 0],
        ],
        [
            [datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 1)],
            [datetime(1970, 1, 4, 11, 20, 3), datetime(1970, 1, 4, 11, 20, 3)],
            [datetime(1970, 1, 1, 0, 0, 0), datetime(1970, 1, 1, 0, 0, 0)],
        ],
        ["float_", "int_"],
    ),
    (
        [
            [datetime(2021, 8, 22), "2021-08-22", "22.08.2021"],
            [
                datetime(2021, 8, 23, 11, 29, 29),
                "2021-08-23 11:29:29",
                "23.08.2021-11:29:29",
            ],
            [datetime(2021, 8, 27), "2021-08-27", "27.08.2021"],
        ],
        [
            [
                datetime(2021, 8, 22),
                datetime(2021, 8, 22),
                datetime(2021, 8, 22),
            ],
            [
                datetime(2021, 8, 23, 11, 29, 29),
                datetime(2021, 8, 23, 11, 29, 29),
                datetime(2021, 8, 23, 11, 29, 29),
            ],
            [
                datetime(2021, 8, 27),
                datetime(2021, 8, 27),
                datetime(2021, 8, 27),
            ],
        ],
        ["ts_", "str_", "str_format_"],
    ),
]


@pytest.mark.parametrize(
    "log_data, ground_truth_data, schema", different_timestamp_formats_data
)
def test_process_timestamp(log_data, ground_truth_data, schema, spark):
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    log = spark.createDataFrame(data=log_data, schema=schema)
    ground_truth = spark.createDataFrame(data=ground_truth_data, schema=schema)
    for col in log.columns:
        kwargs = (
            {"date_format": "dd.MM.yyyy[-HH:mm:ss[.SSS]]"}
            if col == "str_format_"
            else {}
        )
        log = utils.process_timestamp_column(log, col, **kwargs)
        assert isinstance(log.schema[col].dataType, TimestampType)
    sparkDataFrameEqual(log, ground_truth)
    spark.conf.unset("spark.sql.session.timeZone")


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
