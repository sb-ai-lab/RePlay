import pytest


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
