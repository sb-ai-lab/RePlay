from os.path import dirname, join

import pandas as pd
import polars as pl
import pytest

import replay
from replay.data.dataset import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType

cols = ["user_id", "item_id", "rating"]

data = [
    [1, 1, 0.5],
    [
        1,
        2,
        1.0,
    ],
    # TODO: Use a lot dfs - empty, not sorted, one user/item_id, one None item / user, check conftest
    [2, 2, 0.1],
    [2, 3, 0.8],
    [3, 3, 0.7],
    [4, 3, 1.0],
]

new_data = [[5, 4, 1.0]]
feature_schema_small_df = FeatureSchema(
    [
        FeatureInfo(
            column="user_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.QUERY_ID,
        ),
        FeatureInfo(
            column="item_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
        ),
        FeatureInfo(
            column="rating",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.RATING,
        ),
    ]
)

feature_schema = FeatureSchema(
    [
        FeatureInfo(
            column="user_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.QUERY_ID,
        ),
        FeatureInfo(
            column="item_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
        ),
        FeatureInfo(
            column="rating",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.RATING,
        ),
        FeatureInfo(
            column="timestamp",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.TIMESTAMP,
        ),
    ]
)


@pytest.fixture(scope="module")
def cat_tree(spark):
    return spark.createDataFrame(
        data=[
            [None, "healthy_food"],
            [None, "groceries"],
            ["groceries", "fruits"],
            ["fruits", "apples"],
            ["fruits", "bananas"],
            ["apples", "red_apples"],
        ],
        schema="parent_cat string, category string",
    )


@pytest.fixture(scope="module")
def cat_log(spark):
    # assume item 1 is an apple-banana mix and item 2 is a banana
    return spark.createDataFrame(
        data=[
            [1, 1, "red_apples", 5],
            [1, 2, "bananas", 1],
            [2, 1, "healthy_food", 3],
            [3, 1, "bananas", 2],
        ],
        schema="user_idx int, item_idx int, category string, relevance int",
    )


@pytest.fixture(scope="module")
def requested_cats(spark):
    return spark.createDataFrame(
        data=[
            ["healthy_food"],
            ["fruits"],
            ["red_apples"],
        ],
        schema="category string",
    )


@pytest.fixture(scope="module")
def pandas_interactions():
    return pd.DataFrame(data, columns=cols)


@pytest.fixture(scope="module")
def spark_interactions(spark, pandas_interactions):
    return spark.createDataFrame(pandas_interactions)


@pytest.fixture(scope="module")
def polars_interactions(pandas_interactions):
    return pl.DataFrame(pandas_interactions)


@pytest.fixture(scope="function")
def datasets(spark_interactions, polars_interactions, pandas_interactions):
    return {
        "pandas": Dataset(feature_schema_small_df, pandas_interactions),
        "polars": Dataset(feature_schema_small_df, polars_interactions),
        "spark": Dataset(feature_schema_small_df, spark_interactions),
    }


@pytest.fixture(scope="function")
def pandas_big_df():
    folder = dirname(replay.__file__)
    res = pd.read_csv(
        join(folder, "../examples/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    ).head(1000)
    return res


@pytest.fixture(scope="function")
def polars_big_df(pandas_big_df):
    return pl.from_pandas(pandas_big_df)


@pytest.fixture(scope="function")
def spark_big_df(spark, pandas_big_df):
    return spark.createDataFrame(pandas_big_df)


@pytest.fixture(scope="function")
def big_datasets(pandas_big_df, polars_big_df, spark_big_df):
    return {
        "pandas": Dataset(feature_schema, pandas_big_df),
        "polars": Dataset(feature_schema, polars_big_df),
        "spark": Dataset(feature_schema, spark_big_df),
    }
