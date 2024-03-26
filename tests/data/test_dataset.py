import pandas as pd
import polars as pl
import pytest

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType
from replay.utils import PYSPARK_AVAILABLE

if PYSPARK_AVAILABLE:
    from pyspark.sql.functions import asc
    from pyspark.storagelevel import StorageLevel


def get_features(data_dict):
    users = data_dict.get("users", None)
    items = data_dict.get("items", None)
    timestamp_col = data_dict.get("timestamp_col", None)
    ratings_col = data_dict.get("ratings_col", None)

    features = [
        FeatureInfo(
            column=data_dict["user_col"],
            feature_hint=FeatureHint.QUERY_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
        FeatureInfo(
            column=data_dict["item_col"],
            feature_hint=FeatureHint.ITEM_ID,
            feature_type=FeatureType.CATEGORICAL,
            cardinality=data_dict["items_cardinality"],
        ),
    ]

    if timestamp_col:
        features += [
            FeatureInfo(
                column=timestamp_col,
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.TIMESTAMP,
            )
        ]

    if ratings_col:
        features += [
            FeatureInfo(
                column=ratings_col,
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.RATING,
            )
        ]

    if users is not None:
        features += [
            FeatureInfo(
                column="gender",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    if items is not None:
        features += [
            FeatureInfo(
                column="category_id",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    return FeatureSchema(features)


def get_wrong_users_features(data_dict):
    users = data_dict.get("users", None)
    items = data_dict.get("items", None)

    features = [
        FeatureInfo(
            column=data_dict["user_col"],
            feature_hint=FeatureHint.QUERY_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
        FeatureInfo(
            column=data_dict["user_col2"],
            feature_hint=FeatureHint.QUERY_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
        FeatureInfo(
            column=data_dict["item_col"],
            feature_hint=FeatureHint.ITEM_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
    ]
    if users is not None:
        features += [
            FeatureInfo(
                column="gender",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    if items is not None:
        features += [
            FeatureInfo(
                column="category_id",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    return FeatureSchema(features)


def get_wrong_items_features(data_dict):
    users = data_dict.get("users", None)
    items = data_dict.get("items", None)

    features = [
        FeatureInfo(
            column=data_dict["user_col"],
            feature_hint=FeatureHint.QUERY_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
        FeatureInfo(
            column=data_dict["item_col2"],
            feature_hint=FeatureHint.ITEM_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
        FeatureInfo(
            column=data_dict["item_col"],
            feature_hint=FeatureHint.ITEM_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
    ]
    if users is not None:
        features += [
            FeatureInfo(
                column="gender",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    if items is not None:
        features += [
            FeatureInfo(
                column="category_id",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    return FeatureSchema(features)


def get_doesnt_existing_features(data_dict):
    users = data_dict.get("users", None)
    items = data_dict.get("items", None)

    features = [
        FeatureInfo(
            column=data_dict["user_col"],
            feature_hint=FeatureHint.QUERY_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
        FeatureInfo(
            column=data_dict["item_col"],
            feature_hint=FeatureHint.ITEM_ID,
            feature_type=FeatureType.CATEGORICAL,
        ),
        FeatureInfo(
            column="fake",
            feature_type=FeatureType.CATEGORICAL,
        ),
    ]
    if users is not None:
        features += [
            FeatureInfo(
                column="gender",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    if items is not None:
        features += [
            FeatureInfo(
                column="category_id",
                feature_type=FeatureType.CATEGORICAL,
            )
        ]

    return FeatureSchema(features)


def create_dataset(data_dict, check_consistency=True, categorical_encoded=True):
    features = get_features(data_dict)
    return Dataset(
        feature_schema=features,
        interactions=data_dict["interactions"],
        query_features=data_dict.get("users", None),
        item_features=data_dict.get("items", None),
        check_consistency=check_consistency,
        categorical_encoded=categorical_encoded,
    )


def compare_storage_level(resulting_storage_level, expected_storage_level):
    assert resulting_storage_level.useDisk == expected_storage_level.useDisk
    assert resulting_storage_level.useMemory == expected_storage_level.useMemory
    assert resulting_storage_level.useOffHeap == expected_storage_level.useOffHeap
    assert resulting_storage_level.deserialized == expected_storage_level.deserialized
    assert resulting_storage_level.replication == expected_storage_level.replication


@pytest.mark.spark
@pytest.mark.parametrize(
    "data_dict",
    [
        ("wrong_item_pandas_dataset"),
        ("wrong_user_pandas_dataset"),
    ],
)
def test_wrong_features_type(data_dict, request):
    with pytest.raises(TypeError):
        create_dataset(request.getfixturevalue(data_dict))


@pytest.mark.core
@pytest.mark.usefixtures("interactions_full_pandas_dataset")
def test_type_pandas(interactions_full_pandas_dataset):
    dataset = create_dataset(interactions_full_pandas_dataset, check_consistency=True, categorical_encoded=False)
    assert dataset.is_pandas
    assert not dataset.is_spark
    assert not dataset.is_polars
    assert dataset.is_categorical_encoded is False


@pytest.mark.core
@pytest.mark.usefixtures("interactions_full_polars_dataset")
def test_type_polars(interactions_full_polars_dataset):
    dataset = create_dataset(interactions_full_polars_dataset, check_consistency=True, categorical_encoded=False)
    assert dataset.is_polars
    assert not dataset.is_pandas
    assert not dataset.is_spark
    assert dataset.is_categorical_encoded is False


@pytest.mark.spark
@pytest.mark.usefixtures("interactions_full_spark_dataset")
def test_type_spark(interactions_full_spark_dataset):
    dataset = create_dataset(interactions_full_spark_dataset)
    assert not dataset.is_pandas
    assert not dataset.is_polars
    assert dataset.is_spark


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_consistent_ids(data_dict, request):
    create_dataset(request.getfixturevalue(data_dict))


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("inconsistent_item_full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("inconsistent_user_full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("inconsistent_item_full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("inconsistent_user_full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("inconsistent_item_full_polars_dataset", marks=pytest.mark.core),
        pytest.param("inconsistent_user_full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_inconsistent_ids(data_dict, request):
    with pytest.raises(ValueError):
        create_dataset(request.getfixturevalue(data_dict))


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("not_int_user_spark_dataset", marks=pytest.mark.spark),
        pytest.param("not_int_item_spark_dataset", marks=pytest.mark.spark),
        pytest.param("less_than_zero_user_spark_dataset", marks=pytest.mark.spark),
        pytest.param("less_than_zero_item_spark_dataset", marks=pytest.mark.spark),
        pytest.param("more_than_count_user_spark_dataset", marks=pytest.mark.spark),
        pytest.param("more_than_count_item_spark_dataset", marks=pytest.mark.spark),
        pytest.param("not_int_user_pandas_dataset", marks=pytest.mark.core),
        pytest.param("not_int_item_pandas_dataset", marks=pytest.mark.core),
        pytest.param("less_than_zero_user_pandas_dataset", marks=pytest.mark.core),
        pytest.param("less_than_zero_item_pandas_dataset", marks=pytest.mark.core),
        pytest.param("more_than_count_user_pandas_dataset", marks=pytest.mark.core),
        pytest.param("more_than_count_item_pandas_dataset", marks=pytest.mark.core),
        pytest.param("not_int_user_polars_dataset", marks=pytest.mark.core),
        pytest.param("not_int_item_polars_dataset", marks=pytest.mark.core),
        pytest.param("less_than_zero_user_polars_dataset", marks=pytest.mark.core),
        pytest.param("less_than_zero_item_polars_dataset", marks=pytest.mark.core),
        pytest.param("more_than_count_user_polars_dataset", marks=pytest.mark.core),
        pytest.param("more_than_count_item_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_unencoded_ids(data_dict, request):
    with pytest.raises(ValueError):
        create_dataset(request.getfixturevalue(data_dict))


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("inconsistent_item_full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("not_int_user_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("inconsistent_user_full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("not_int_user_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
        pytest.param("inconsistent_user_full_polars_dataset", marks=pytest.mark.core),
        pytest.param("not_int_user_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_not_check_consistency(data_dict, request):
    create_dataset(request.getfixturevalue(data_dict), check_consistency=False)


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_get_unlabeled_columns(data_dict, request):
    feature_schema = get_features(request.getfixturevalue(data_dict))
    dataset = create_dataset(request.getfixturevalue(data_dict))
    unlabeled_cols = dataset._get_unlabeled_columns(source=FeatureSource.ITEM_FEATURES, feature_schema=feature_schema)
    assert len(unlabeled_cols) == 1
    assert unlabeled_cols[0].column == "feature1"


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_info_doesnt_exist(data_dict, request):
    dataframe = request.getfixturevalue(data_dict)
    feature_schema = get_doesnt_existing_features(dataframe)

    with pytest.raises(ValueError) as exc:
        Dataset(
            feature_schema=feature_schema,
            interactions=dataframe["interactions"],
            query_features=dataframe.get("users", None),
            item_features=dataframe.get("items", None),
            check_consistency=True,
            categorical_encoded=False,
        )

    assert str(exc.value) == "fake doesn't exist in provided dataframes"


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_fill_feature_info_dunder(data_dict, request):
    feature_schema = get_features(request.getfixturevalue(data_dict))
    dataset = create_dataset(request.getfixturevalue(data_dict))
    feature_schema_updated = dataset.feature_schema
    assert feature_schema_updated.get("feature1") is not None
    assert feature_schema_updated["feature1"] is not None
    assert feature_schema_updated == feature_schema_updated
    assert feature_schema_updated != feature_schema
    assert "feature1" in feature_schema_updated
    assert "feature1" in feature_schema_updated.keys()
    assert next(iter(feature_schema_updated)) is not None


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_fill_feature_schema(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert "feature1" in dataset.feature_schema.columns


@pytest.mark.parametrize(
    "data_dict, answer",
    [
        pytest.param("full_spark_dataset", 4, marks=pytest.mark.spark),
        pytest.param("full_spark_dataset_cutted_interactions", 4, marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", 4, marks=pytest.mark.core),
        pytest.param("full_pandas_dataset_cutted_interactions", 4, marks=pytest.mark.core),
        pytest.param("full_polars_dataset", 4, marks=pytest.mark.core),
        pytest.param("full_polars_dataset_cutted_interactions", 4, marks=pytest.mark.core),
    ],
)
def test_item_count(data_dict, answer, request):
    dataset = create_dataset(request.getfixturevalue(data_dict), check_consistency=False)
    assert dataset.item_count == answer


@pytest.mark.parametrize(
    "data_dict, answer",
    [
        pytest.param("full_spark_dataset", 3, marks=pytest.mark.spark),
        pytest.param("full_spark_dataset_cutted_interactions", 3, marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", 3, marks=pytest.mark.core),
        pytest.param("full_pandas_dataset_cutted_interactions", 3, marks=pytest.mark.core),
        pytest.param("full_polars_dataset", 3, marks=pytest.mark.core),
        pytest.param("full_polars_dataset_cutted_interactions", 3, marks=pytest.mark.core),
    ],
)
def test_user_count(data_dict, answer, request):
    dataset = create_dataset(request.getfixturevalue(data_dict), check_consistency=False)
    assert dataset.query_count == answer


@pytest.mark.spark
@pytest.mark.parametrize(
    "data_dict",
    [
        ("full_spark_dataset"),
        ("interactions_full_spark_dataset"),
        ("interactions_users_spark_dataset"),
        ("interactions_items_spark_dataset"),
    ],
)
def test_persist_spark(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    dataset.persist(StorageLevel.MEMORY_ONLY)
    compare_storage_level(dataset.interactions.storageLevel, StorageLevel.MEMORY_ONLY)
    if dataset.query_features:
        compare_storage_level(dataset.query_features.storageLevel, StorageLevel.MEMORY_ONLY)
    if dataset.item_features:
        compare_storage_level(dataset.item_features.storageLevel, StorageLevel.MEMORY_ONLY)


@pytest.mark.spark
@pytest.mark.parametrize(
    "data_dict",
    [
        ("full_spark_dataset"),
        ("interactions_full_spark_dataset"),
        ("interactions_users_spark_dataset"),
        ("interactions_items_spark_dataset"),
    ],
)
def test_unpersist_spark(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    dataset.persist(StorageLevel.MEMORY_ONLY)
    dataset.unpersist()

    compare_storage_level(dataset.interactions.storageLevel, StorageLevel(False, False, False, False, 1))
    if dataset.query_features:
        compare_storage_level(dataset.query_features.storageLevel, StorageLevel(False, False, False, False, 1))
    if dataset.item_features:
        compare_storage_level(dataset.item_features.storageLevel, StorageLevel(False, False, False, False, 1))


@pytest.mark.spark
@pytest.mark.parametrize(
    "data_dict",
    [
        ("full_spark_dataset"),
        ("interactions_full_spark_dataset"),
        ("interactions_users_spark_dataset"),
        ("interactions_items_spark_dataset"),
    ],
)
def test_cache_spark(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    dataset.cache()
    compare_storage_level(dataset.interactions.storageLevel, StorageLevel(True, True, False, True, 1))
    if dataset.query_features:
        compare_storage_level(dataset.query_features.storageLevel, StorageLevel(True, True, False, True, 1))
    if dataset.item_features:
        compare_storage_level(dataset.item_features.storageLevel, StorageLevel(True, True, False, True, 1))


@pytest.mark.parametrize(
    "data_dict, source, answer",
    [
        pytest.param("full_spark_dataset", FeatureHint.ITEM_ID, "item_features", marks=pytest.mark.spark),
        pytest.param("full_spark_dataset", FeatureSource.ITEM_FEATURES, "item_features", marks=pytest.mark.spark),
        pytest.param("full_spark_dataset", FeatureHint.QUERY_ID, "query_features", marks=pytest.mark.spark),
        pytest.param("full_spark_dataset", FeatureSource.QUERY_FEATURES, "query_features", marks=pytest.mark.spark),
        pytest.param("full_spark_dataset", FeatureSource.INTERACTIONS, "interactions", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", FeatureHint.ITEM_ID, "item_features", marks=pytest.mark.core),
        pytest.param("full_pandas_dataset", FeatureSource.ITEM_FEATURES, "item_features", marks=pytest.mark.core),
        pytest.param("full_pandas_dataset", FeatureHint.QUERY_ID, "query_features", marks=pytest.mark.core),
        pytest.param("full_pandas_dataset", FeatureSource.QUERY_FEATURES, "query_features", marks=pytest.mark.core),
        pytest.param("full_pandas_dataset", FeatureSource.INTERACTIONS, "interactions", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", FeatureHint.ITEM_ID, "item_features", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", FeatureSource.ITEM_FEATURES, "item_features", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", FeatureHint.QUERY_ID, "query_features", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", FeatureSource.QUERY_FEATURES, "query_features", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", FeatureSource.INTERACTIONS, "interactions", marks=pytest.mark.core),
    ],
)
def test_dataframe_by_source(data_dict, source, answer, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    if isinstance(source, FeatureSource):
        assert id(dataset._feature_source_map[source]) == id(getattr(dataset, answer))
    else:
        assert id(dataset._ids_feature_map[source]) == id(getattr(dataset, answer))


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_dict(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.items() is not None
    assert dataset.feature_schema.values() is not None
    assert dataset.feature_schema.keys() is not None


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_all_features(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert len(dataset.feature_schema.all_features) == 7


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_timestamp_column(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.interactions_timestamp_column == "timestamp"


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_rating_column(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.interactions_rating_column == "rating"


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_user_id_column(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.query_id_column == "user_id"


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_item_id_column(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.item_id_column == "item_id"


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_columns(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.columns == [
        "user_id",
        "item_id",
        "timestamp",
        "rating",
        "gender",
        "category_id",
        "feature1",
    ]


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_categorical_features(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.categorical_features.columns == [
        "user_id",
        "item_id",
        "timestamp",
        "gender",
        "category_id",
    ]


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_numerical_features(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.numerical_features.columns == ["rating", "feature1"]


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_item_features(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.item_features.columns == ["category_id", "feature1"]


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_user_features(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.query_features.columns == ["gender"]


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("full_spark_dataset", marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("full_polars_dataset", marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_interaction_features(data_dict, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert dataset.feature_schema.interaction_features.columns == ["timestamp", "rating"]


@pytest.mark.parametrize(
    "data_dict, name, source, type, hint, result",
    [
        pytest.param("full_spark_dataset", "gender", None, None, None, ["gender"], marks=pytest.mark.spark),
        pytest.param(
            "full_spark_dataset",
            "feature1",
            FeatureSource.ITEM_FEATURES,
            FeatureType.NUMERICAL,
            None,
            ["feature1"],
            marks=pytest.mark.spark,
        ),
        pytest.param(
            "full_spark_dataset",
            None,
            FeatureSource.ITEM_FEATURES,
            None,
            None,
            ["category_id", "feature1"],
            marks=pytest.mark.spark,
        ),
        pytest.param("full_pandas_dataset", "gender", None, None, None, ["gender"], marks=pytest.mark.core),
        pytest.param(
            "full_pandas_dataset",
            "feature1",
            FeatureSource.ITEM_FEATURES,
            FeatureType.NUMERICAL,
            None,
            ["feature1"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_pandas_dataset",
            None,
            FeatureSource.ITEM_FEATURES,
            None,
            None,
            ["category_id", "feature1"],
            marks=pytest.mark.core,
        ),
        pytest.param("full_polars_dataset", "gender", None, None, None, ["gender"], marks=pytest.mark.core),
        pytest.param(
            "full_polars_dataset",
            "feature1",
            FeatureSource.ITEM_FEATURES,
            FeatureType.NUMERICAL,
            None,
            ["feature1"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_polars_dataset",
            None,
            FeatureSource.ITEM_FEATURES,
            None,
            None,
            ["category_id", "feature1"],
            marks=pytest.mark.core,
        ),
    ],
)
def test_feature_schema_schema_filter(data_dict, name, source, type, hint, result, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert (
        dataset.feature_schema.filter(column=name, feature_source=source, feature_type=type, feature_hint=hint).columns
        == result
    )


@pytest.mark.parametrize(
    "data_dict, name, source, type, hint, result",
    [
        pytest.param(
            "full_spark_dataset",
            "gender",
            None,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "category_id", "feature1"],
            marks=pytest.mark.spark,
        ),
        pytest.param(
            "full_spark_dataset",
            "feature1",
            FeatureSource.ITEM_FEATURES,
            FeatureType.NUMERICAL,
            None,
            ["user_id", "item_id", "timestamp", "gender"],
            marks=pytest.mark.spark,
        ),
        pytest.param(
            "full_spark_dataset",
            None,
            FeatureSource.ITEM_FEATURES,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "gender"],
            marks=pytest.mark.spark,
        ),
        pytest.param(
            "full_pandas_dataset",
            "gender",
            None,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "category_id", "feature1"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_pandas_dataset",
            "feature1",
            FeatureSource.ITEM_FEATURES,
            FeatureType.NUMERICAL,
            None,
            ["user_id", "item_id", "timestamp", "gender"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_pandas_dataset",
            None,
            FeatureSource.ITEM_FEATURES,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "gender"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_polars_dataset",
            "gender",
            None,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "category_id", "feature1"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_polars_dataset",
            "feature1",
            FeatureSource.ITEM_FEATURES,
            FeatureType.NUMERICAL,
            None,
            ["user_id", "item_id", "timestamp", "gender"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_polars_dataset",
            None,
            FeatureSource.ITEM_FEATURES,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "gender"],
            marks=pytest.mark.core,
        ),
    ],
)
def test_feature_schema_schema_drop(data_dict, name, source, type, hint, result, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert (
        dataset.feature_schema.drop(column=name, feature_source=source, feature_type=type, feature_hint=hint).columns
        == result
    )


@pytest.mark.parametrize(
    "data_dict, name, source, type, hint",
    [
        pytest.param("full_spark_dataset", "gender100", None, None, None, marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", "gender100", None, None, None, marks=pytest.mark.core),
        pytest.param("full_polars_dataset", "gender100", None, None, None, marks=pytest.mark.core),
    ],
)
def test_feature_schema_schema_filter_empty(data_dict, name, source, type, hint, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert (
        len(
            dataset.feature_schema.filter(
                column=name,
                feature_source=source,
                feature_type=type,
                feature_hint=hint,
            )
        )
        == 0
    )


@pytest.mark.parametrize(
    "data_dict, name, source, type, hint, answer",
    [
        pytest.param(
            "full_spark_dataset",
            "gender100",
            None,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "gender", "category_id", "feature1"],
            marks=pytest.mark.spark,
        ),
        pytest.param(
            "full_pandas_dataset",
            "gender100",
            None,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "gender", "category_id", "feature1"],
            marks=pytest.mark.core,
        ),
        pytest.param(
            "full_polars_dataset",
            "gender100",
            None,
            None,
            None,
            ["user_id", "item_id", "timestamp", "rating", "gender", "category_id", "feature1"],
            marks=pytest.mark.core,
        ),
    ],
)
def test_feature_schema_schema_drop_empty(data_dict, name, source, type, hint, answer, request):
    dataset = create_dataset(request.getfixturevalue(data_dict))
    assert (
        dataset.feature_schema.drop(
            column=name,
            feature_source=source,
            feature_type=type,
            feature_hint=hint,
        ).columns
        == answer
    )


@pytest.mark.core
def test_feature_info_invalid_initialization():
    with pytest.raises(ValueError) as exc:
        FeatureInfo("", FeatureType.NUMERICAL, cardinality=4)

    assert str(exc.value) == "Cardinality is needed only with categorical feature_type."


@pytest.mark.core
def test_feature_info_numerical_type_cardinality_exception():
    with pytest.raises(RuntimeError) as exc:
        fi = FeatureInfo("", FeatureType.CATEGORICAL, cardinality=4)
        fi._feature_type = FeatureType.NUMERICAL
        fi.cardinality

    assert str(exc.value) == "Can not get cardinality because feature_type of  column is not categorical."


@pytest.mark.core
def test_reset_feature_info_cardinality():
    fi = FeatureInfo("", FeatureType.CATEGORICAL, cardinality=4)
    fi.reset_cardinality()

    assert fi.cardinality is None


@pytest.mark.core
@pytest.mark.usefixtures("interactions_full_pandas_dataset")
def test_feature_schema_schema_copy(interactions_full_pandas_dataset):
    feature_list = get_features(interactions_full_pandas_dataset)
    feature_list_copy = feature_list.copy()

    for feature in feature_list_copy.values():
        if feature.feature_type == FeatureType.CATEGORICAL:
            assert feature.cardinality is None

    assert bool(feature_list_copy) is True
    assert len(feature_list_copy) == len(feature_list)
    assert len(feature_list_copy.subset(["user_id"])) == len(feature_list.subset(["user_id"]))


@pytest.mark.core
@pytest.mark.parametrize(
    "data_dict, feature_schema_getter, error_msg",
    [
        pytest.param(
            "full_pandas_dataset_nonunique_columns",
            get_features,
            (
                "Features column names should be unique, exept ITEM_ID and QUERY_ID columns. "
                + "{'feature1'} columns are not unique."
            ),
        ),
        pytest.param(
            "users_pandas_dataset_different_columns",
            get_wrong_users_features,
            "QUERY_ID must be present only once. Rename ['user_id', 'user_ids']",
        ),
        pytest.param(
            "items_pandas_dataset_different_columns",
            get_wrong_items_features,
            "ITEM_ID must be present only once. Rename ['item_ids', 'item_id']",
        ),
    ],
)
def test_feature_schema_check_naming(data_dict, feature_schema_getter, error_msg, request):
    dataframe = request.getfixturevalue(data_dict)
    with pytest.raises(ValueError) as exc:
        feature_schema = feature_schema_getter(dataframe)
        Dataset(
            feature_schema=feature_schema,
            interactions=dataframe["interactions"],
            query_features=dataframe.get("users", None),
            item_features=dataframe.get("items", None),
            check_consistency=True,
            categorical_encoded=False,
        )

    assert str(exc.value) == error_msg


@pytest.mark.core
@pytest.mark.usefixtures("interactions_full_pandas_dataset")
def test_feature_schema_schema_item_error(interactions_full_pandas_dataset):
    with pytest.raises(ValueError) as exc:
        get_features(interactions_full_pandas_dataset).item()

    assert str(exc.value) == "Only one element feature schema can be converted to single feature"


@pytest.mark.core
@pytest.mark.usefixtures("interactions_full_pandas_dataset")
def test_feature_schema_schema_empty_properties(interactions_full_pandas_dataset):
    feature_list = get_features(interactions_full_pandas_dataset).subset(["user_id"])

    assert feature_list.interactions_rating_column is None
    assert feature_list.interactions_timestamp_column is None


@pytest.mark.core
@pytest.mark.parametrize(
    "subset, error_msg",
    [
        (["timestamp"], "Item id column is not set."),
        (["rating", "item_id"], "Query id column is not set."),
    ],
)
@pytest.mark.usefixtures("interactions_full_pandas_dataset")
def test_interactions_dataset_init_exceptions(interactions_full_pandas_dataset, subset, error_msg):
    with pytest.raises(ValueError) as exc:
        Dataset(
            feature_schema=get_features(interactions_full_pandas_dataset).subset(subset),
            interactions=interactions_full_pandas_dataset["interactions"],
            query_features=interactions_full_pandas_dataset.get("users", None),
            item_features=interactions_full_pandas_dataset.get("items", None),
            check_consistency=True,
            categorical_encoded=False,
        )

    assert str(exc.value) == error_msg


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("interactions_full_pandas_dataset", marks=pytest.mark.core),
        pytest.param("interactions_full_polars_dataset", marks=pytest.mark.core),
        pytest.param("interactions_full_spark_dataset", marks=pytest.mark.spark),
    ],
)
def test_interaction_dataset_queryids_and_itemids(dataset, request):
    dataset = create_dataset(request.getfixturevalue(dataset))

    if dataset.is_spark:
        assert [x.user_id for x in dataset.query_ids.sort(asc("user_id")).collect()] == [0, 1, 2]
        assert [x.item_id for x in dataset.item_ids.sort(asc("item_id")).collect()] == [0, 1, 2, 3]
    else:
        assert sorted(dataset.query_ids["user_id"].to_list()) == [0, 1, 2]
        assert sorted(dataset.item_ids["item_id"].to_list()) == [0, 1, 2, 3]


@pytest.mark.core
@pytest.mark.parametrize(
    "users, items, subset, columns_len",
    [
        (
            pd.DataFrame({"user_id": [0, 1, 2], "gender": [0, 1, 0]}),
            None,
            [
                "user_id",
                "item_id",
                "gender",
                "gender_fake",
            ],
            3,
        ),
        (
            None,
            pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]}),
            [
                "user_id",
                "item_id",
                "category_id",
            ],
            3,
        ),
        (
            pd.DataFrame({"user_id": [0, 1, 2], "gender": [0, 1, 0]}),
            pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]}),
            [
                "user_id",
                "item_id",
                "gender",
                "category_id",
            ],
            4,
        ),
    ],
)
def test_interactions_dataset_subset(full_pandas_dataset, users, items, subset, columns_len):
    full_pandas_dataset["users"] = users
    full_pandas_dataset["items"] = items
    dataset = create_dataset(full_pandas_dataset)
    dataset_subset = dataset.subset(subset)
    columns = dataset_subset.interactions.columns.to_list()
    if users is not None:
        columns += dataset_subset.query_features.columns.to_list()
    if items is not None:
        columns += dataset_subset.item_features.columns.to_list()

    assert dataset_subset.is_pandas is True
    assert len(set(columns)) == columns_len


@pytest.mark.core
@pytest.mark.parametrize(
    "users, items, subset, columns_len",
    [
        (
            pl.DataFrame({"user_id": [0, 1, 2], "gender": [0, 1, 0]}),
            None,
            [
                "user_id",
                "item_id",
                "gender",
                "gender_fake",
            ],
            3,
        ),
        (
            None,
            pl.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]}),
            [
                "user_id",
                "item_id",
                "category_id",
            ],
            3,
        ),
        (
            pl.DataFrame({"user_id": [0, 1, 2], "gender": [0, 1, 0]}),
            pl.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]}),
            [
                "user_id",
                "item_id",
                "gender",
                "category_id",
            ],
            4,
        ),
    ],
)
def test_interactions_dataset_subset_polars(full_polars_dataset, users, items, subset, columns_len):
    full_polars_dataset["users"] = users
    full_polars_dataset["items"] = items
    dataset = create_dataset(full_polars_dataset)
    dataset_subset = dataset.subset(subset)
    columns = dataset_subset.interactions.columns
    if users is not None:
        columns += dataset_subset.query_features.columns
    if items is not None:
        columns += dataset_subset.item_features.columns

    assert dataset_subset.is_polars is True
    assert len(set(columns)) == columns_len


@pytest.mark.spark
def test_pandas_dataframe_in_storage_levels_of_spark(full_pandas_dataset):
    dataset = create_dataset(full_pandas_dataset)

    assert dataset.persist() is None
    assert dataset.unpersist() is None
    assert dataset.cache() is None
