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


def test_join_based_indexer(spark, log):
    log.show()
    indexer = JoinBasedIndexerEstimator().fit(log)
    indexed_df = indexer.transform(log)
    indexed_df.show()
    assert (
        "user_idx" in indexed_df.columns and "item_idx" in indexed_df.columns
    )
    assert log.count() == indexed_df.count()
    assert indexed_df.agg({"user_idx": "max"}).collect()[0][0] < log.count()
    assert indexed_df.agg({"item_idx": "max"}).collect()[0][0] < log.count()
    assert (
        indexed_df.select(sf.min("user_idx") + sf.min("item_idx")).collect()[
            0
        ][0]
        == 0
    )

    # inverse indexing
    df_with_primary_indexes = indexer.inverse_transform(indexed_df)

    # check that the number of rows remains the same
    assert indexed_df.count() == df_with_primary_indexes.count()

    # check the inverse indexing for users
    expected_unique_user_ids = (
        log.select("user_id").distinct().sort("user_id").collect()
    )
    actual_unique_user_ids = (
        df_with_primary_indexes.select("user_id")
        .distinct()
        .sort("user_id")
        .collect()
    )
    assert expected_unique_user_ids == actual_unique_user_ids

    # check the inverse indexing for items
    expected_unique_item_ids = (
        log.select("item_id").distinct().sort("item_id").collect()
    )
    actual_unique_item_ids = (
        df_with_primary_indexes.select("item_id")
        .distinct()
        .sort("item_id")
        .collect()
    )
    assert expected_unique_item_ids == actual_unique_item_ids

    # check indexing new user-items
    indexer.set_update_map_on_transform(True)

    # add new user and two new items
    new_log_part = spark.createDataFrame([(100, 1), (100, 2)]).toDF(
        "user_id", "item_id"
    )
    new_log = log.union(new_log_part)

    new_indexed_df = indexer.transform(new_log)
    new_indexed_df.show()

    # check that the number of unique users and items remains the same
    assert (
        new_log.select("user_id").distinct().count()
        == new_indexed_df.select("user_idx").distinct().count()
    )
    assert (
        new_log.select("item_id").distinct().count()
        == new_indexed_df.select("item_idx").distinct().count()
    )

    # check that new indexed dataframe contains indexes from previous indexed dataframe
    assert (
        indexed_df.join(new_indexed_df, on=["user_idx", "item_idx"]).count()
        == indexed_df.count()
    )
    assert (
        indexed_df.join(new_indexed_df, on=["user_idx", "item_idx"])
        .distinct()
        .count()
        == indexed_df.distinct().count()
    )

    # check indexing of new elements
    indexed_new_log_part_df = new_indexed_df.join(
        indexed_df, on=["user_idx", "item_idx"], how="left_anti"
    )
    indexed_new_log_part_df.show()
    assert (
        indexed_new_log_part_df.select("user_idx").distinct().count()
        == new_log_part.select("user_id").distinct().count()
    )
    assert (
        indexed_new_log_part_df.select("item_idx").distinct().count()
        == new_log_part.select("item_id").distinct().count()
    )
