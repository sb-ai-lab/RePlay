# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, wildcard-import, unused-wildcard-import

import pytest
from pyspark.sql import functions as sf

from replay.data_preparator import JoinBasedIndexerEstimator
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        [(55, 70), (56, 70), (57, 71), (88, 72), (55, 72)]
    ).toDF("user_id", "item_id")


def test_join_based_indexer(log):
    log.show()
    indexer = JoinBasedIndexerEstimator().fit(log)
    indexed_df = indexer.transform(log)
    indexed_df.show()
    assert "user_idx" in indexed_df.columns and "item_idx" in indexed_df.columns
    assert log.count() == indexed_df.count()
    assert indexed_df.agg({"user_idx": "max"}).collect()[0][0] < log.count()
    assert indexed_df.agg({"item_idx": "max"}).collect()[0][0] < log.count()
    assert indexed_df.select(sf.min("user_idx") + sf.min("item_idx")).collect()[0][0] == 0
