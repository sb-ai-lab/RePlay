import pandas as pd
import polars as pl
import pytest

from replay.splitters import RandomNextNSplitter
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf


@pytest.fixture(scope="module")
def pandas_dataframe_test():
    columns = ["user_id", "item_id", "timestamp", "session_id"]
    data = [
        (1, 1, "01-01-2020", 1),
        (1, 2, "02-01-2020", 1),
        (1, 3, "03-01-2020", 1),
        (1, 4, "04-01-2020", 1),
        (1, 5, "05-01-2020", 1),
        (2, 1, "06-01-2020", 2),
        (2, 2, "07-01-2020", 2),
        (2, 3, "08-01-2020", 2),
        (2, 9, "09-01-2020", 2),
        (2, 10, "10-01-2020", 2),
        (3, 1, "01-01-2020", 3),
        (3, 5, "02-01-2020", 3),
        (3, 3, "03-01-2020", 3),
        (3, 1, "04-01-2020", 3),
        (3, 2, "05-01-2020", 3),
    ]

    df = pd.DataFrame(data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y")
    return df


@pytest.fixture(scope="module")
def polars_dataframe_test(pandas_dataframe_test):
    return pl.from_pandas(pandas_dataframe_test)


@pytest.fixture(scope="module")
def spark_dataframe_test(spark, pandas_dataframe_test):
    sdf = spark.createDataFrame(
        pandas_dataframe_test.assign(timestamp=pandas_dataframe_test["timestamp"].dt.strftime("%d-%m-%Y"))
    )
    return sdf.withColumn("timestamp", sf.to_date("timestamp", "dd-MM-yyyy"))


@pytest.fixture(scope="module")
def numpy_not_supported(pandas_dataframe_test):
    return pandas_dataframe_test.to_numpy()


SEED = 1234


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_basic_split_n1(dataset_type, request):
    data = request.getfixturevalue(dataset_type)
    splitter = RandomNextNSplitter(
        N=1,
        divide_column="user_id",
        seed=SEED,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
    )
    train, test = splitter.split(data)

    if not isinstance(data, SparkDataFrame):
        if isinstance(data, pd.DataFrame):
            num_users_data = data["user_id"].nunique()
            num_users_test = test["user_id"].nunique()
        else:
            num_users_data = data.select(pl.col("user_id").n_unique()).item()
            num_users_test = test.select(pl.col("user_id").n_unique()).item()
        assert num_users_test == num_users_data

        if isinstance(data, pd.DataFrame):
            original_counts = data.groupby("user_id")["item_id"].size()
            train_counts = train.groupby("user_id")["item_id"].size()
            for user_id, original_len in original_counts.items():
                train_len = int(train_counts.get(user_id, 0))
                assert 0 <= train_len <= (original_len - 1)
        else:
            original_counts = data.group_by("user_id").count().rename({"count": "orig_count"})
            train_counts = train.group_by("user_id").count().rename({"count": "train_count"})
            joined = original_counts.join(train_counts, on="user_id", how="left")
            joined = joined.with_columns(pl.col("train_count").fill_null(0))
            violations = joined.filter(
                (pl.col("train_count") < 0) | (pl.col("train_count") > (pl.col("orig_count") - 1))
            )
            assert violations.height == 0

        cols = list(data.columns)
        if isinstance(data, pd.DataFrame):
            overlap = pd.merge(train, test, on=cols, how="inner")
            assert overlap.shape[0] == 0
        else:
            overlap = train.join(test, on=cols, how="inner")
            assert overlap.height == 0
    else:
        num_users_data = data.select("user_id").distinct().count()
        num_users_test = test.select("user_id").distinct().count()
        assert num_users_test == num_users_data

        original_counts = data.groupBy("user_id").count().withColumnRenamed("count", "orig_count")
        train_counts = train.groupBy("user_id").count().withColumnRenamed("count", "train_count")
        joined = original_counts.join(train_counts, on="user_id", how="left").fillna({"train_count": 0})
        violations = joined.filter(
            (sf.col("train_count") < sf.lit(0)) | (sf.col("train_count") > (sf.col("orig_count") - sf.lit(1)))
        )
        assert violations.count() == 0

        assert train.intersect(test).count() == 0


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_n_none_keeps_all(dataset_type, request):
    data = request.getfixturevalue(dataset_type)
    splitter = RandomNextNSplitter(
        N=None,
        divide_column="user_id",
        seed=SEED,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
    )
    train, test = splitter.split(data)

    if not isinstance(data, SparkDataFrame):
        assert (train.shape[0] + test.shape[0]) == (data.shape[0] if isinstance(data, pd.DataFrame) else data.height)
    else:
        assert (train.count() + test.count()) == data.count()


@pytest.mark.core
def test_invalid_n_raises():
    with pytest.raises(ValueError):
        RandomNextNSplitter(N=0)


@pytest.mark.core
def test_not_implemented_dataframe(numpy_not_supported):
    with pytest.raises(NotImplementedError):
        RandomNextNSplitter(N=1).split(numpy_not_supported)


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_reproducibility_seed_non_spark(dataset_type, request):
    data = request.getfixturevalue(dataset_type)
    splitter_a = RandomNextNSplitter(
        N=2,
        divide_column="user_id",
        seed=SEED,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
    )
    splitter_b = RandomNextNSplitter(
        N=2,
        divide_column="user_id",
        seed=SEED,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
    )
    splitter_c = RandomNextNSplitter(
        N=2,
        divide_column="user_id",
        seed=SEED + 1,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
    )
    train_a, test_a = splitter_a.split(data)
    train_b, test_b = splitter_b.split(data)
    train_c, test_c = splitter_c.split(data)

    if isinstance(data, pd.DataFrame):
        assert train_a.equals(train_b)
        assert test_a.equals(test_b)

        assert not (train_a.equals(train_c) and test_a.equals(test_c))
    else:
        assert train_a.equals(train_b)
        assert test_a.equals(test_b)
        assert not (train_a.equals(train_c) and test_a.equals(test_c))


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_with_session_ids(dataset_type, request):
    data = request.getfixturevalue(dataset_type)

    splitter_train = RandomNextNSplitter(
        N=1,
        divide_column="user_id",
        seed=SEED,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
        session_id_column="session_id",
        session_id_processing_strategy="train",
    )
    splitter_test = RandomNextNSplitter(
        N=1,
        divide_column="user_id",
        seed=SEED,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
        session_id_column="session_id",
        session_id_processing_strategy="test",
    )

    _, test_train = splitter_train.split(data)
    train_test, _ = splitter_test.split(data)

    if not isinstance(data, SparkDataFrame):
        assert (test_train.shape[0] if isinstance(test_train, pd.DataFrame) else test_train.height) == 0
        assert (train_test.shape[0] if isinstance(train_test, pd.DataFrame) else train_test.height) == 0
    else:
        assert test_train.count() == 0
        assert train_test.count() == 0


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_repeated_calls_same_results(dataset_type, request):
    data = request.getfixturevalue(dataset_type)
    splitter = RandomNextNSplitter(
        N=2,
        divide_column="user_id",
        seed=SEED,
        query_column="user_id",
        item_column="item_id",
        timestamp_column="timestamp",
    )

    train_1, test_1 = splitter.split(data)
    train_2, test_2 = splitter.split(data)

    if not isinstance(data, SparkDataFrame):
        assert train_1.equals(train_2)
        assert test_1.equals(test_2)
    else:
        assert train_1.exceptAll(train_2).count() == 0
        assert train_2.exceptAll(train_1).count() == 0
        assert test_1.exceptAll(test_2).count() == 0
        assert test_2.exceptAll(test_1).count() == 0
