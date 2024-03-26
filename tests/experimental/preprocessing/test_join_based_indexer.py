import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.experimental.preprocessing.data_preparator import JoinBasedIndexerEstimator, JoinBasedIndexerTransformer


@pytest.fixture
def log(spark):
    return spark.createDataFrame([(55, 70), (56, 70), (57, 71), (88, 72), (55, 72)]).toDF("user_id", "item_id")


@pytest.mark.experimental
def test_indexer(log):
    indexer = JoinBasedIndexerEstimator().fit(log)
    indexed_df = indexer.transform(log)
    assert "user_idx" in indexed_df.columns and "item_idx" in indexed_df.columns
    assert log.count() == indexed_df.count()
    assert indexed_df.agg({"user_idx": "max"}).collect()[0][0] < log.count()
    assert indexed_df.agg({"item_idx": "max"}).collect()[0][0] < log.count()
    assert indexed_df.select(sf.min("user_idx") + sf.min("item_idx")).collect()[0][0] == 0


@pytest.mark.experimental
def test_inverse_transform(log):
    indexer = JoinBasedIndexerEstimator().fit(log)
    indexed_df = indexer.transform(log)

    # inverse indexing
    df_with_primary_indexes = indexer.inverse_transform(indexed_df)

    # check that the number of rows remains the same
    assert indexed_df.count() == df_with_primary_indexes.count()

    # check the inverse indexing for users
    expected_unique_user_ids = log.select("user_id").distinct().sort("user_id").collect()
    actual_unique_user_ids = df_with_primary_indexes.select("user_id").distinct().sort("user_id").collect()
    assert expected_unique_user_ids == actual_unique_user_ids

    # check the inverse indexing for items
    expected_unique_item_ids = log.select("item_id").distinct().sort("item_id").collect()
    actual_unique_item_ids = df_with_primary_indexes.select("item_id").distinct().sort("item_id").collect()
    assert expected_unique_item_ids == actual_unique_item_ids


@pytest.mark.experimental
def test_update_map_on_transform(spark, log):
    indexer = JoinBasedIndexerEstimator().fit(log)
    indexed_df = indexer.transform(log)

    # add new user and two new items
    new_log_part = spark.createDataFrame([(100, 1), (100, 2)]).toDF("user_id", "item_id")
    new_log = log.union(new_log_part)

    # check indexing new user-items
    new_indexed_df = indexer.transform(new_log)

    # check that the number of unique users and items remains the same
    assert new_log.select("user_id").distinct().count() == new_indexed_df.select("user_idx").distinct().count()
    assert new_log.select("item_id").distinct().count() == new_indexed_df.select("item_idx").distinct().count()

    # check that new indexed dataframe contains indexes from previous indexed dataframe
    assert indexed_df.join(new_indexed_df, on=["user_idx", "item_idx"]).count() == indexed_df.count()
    assert (
        indexed_df.join(new_indexed_df, on=["user_idx", "item_idx"]).distinct().count() == indexed_df.distinct().count()
    )

    # check indexing of new elements
    indexed_new_log_part_df = new_indexed_df.join(indexed_df, on=["user_idx", "item_idx"], how="left_anti")
    assert (
        indexed_new_log_part_df.select("user_idx").distinct().count()
        == new_log_part.select("user_id").distinct().count()
    )
    assert (
        indexed_new_log_part_df.select("item_idx").distinct().count()
        == new_log_part.select("item_id").distinct().count()
    )


@pytest.mark.experimental
def test_save_load(log, tmp_path):
    indexer = JoinBasedIndexerEstimator().fit(log)
    indexed_df_expected = indexer.transform(log)

    path = (tmp_path / "indexer").resolve()
    indexer.save(path)
    loaded_indexer = JoinBasedIndexerTransformer.load(path)

    indexed_df_actual = loaded_indexer.transform(log)
    assert (
        indexed_df_expected.sort(["user_idx", "item_idx"]).collect()
        == indexed_df_actual.sort(["user_idx", "item_idx"]).collect()
    )
