import pytest

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.data.dataset_utils import DatasetLabelEncoder
from replay.preprocessing import LabelEncoder


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
            cardinality=data_dict["users_cardinality"],
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


@pytest.mark.core
@pytest.mark.parametrize(
    "features, is_query_features, is_item_features",
    [
        ([], False, True),
        ([], True, True),
        ([], True, False),
        (["timestamp"], True, True),
        (["gender"], True, True),
        (["category_id"], True, True),
    ],
)
@pytest.mark.usefixtures("full_pandas_dataset")
def test_label_encoder_fit(full_pandas_dataset, features, is_query_features, is_item_features):
    encoder = DatasetLabelEncoder()
    user_item_features = ["user_id", "item_id"]
    dataset = Dataset(
        feature_schema=get_features(full_pandas_dataset).subset(user_item_features + features),
        interactions=full_pandas_dataset["interactions"],
        query_features=full_pandas_dataset["users"] if is_query_features else None,
        item_features=full_pandas_dataset["items"] if is_item_features else None,
    )

    encoder.fit(dataset)
    assert len(encoder._encoding_rules) == len(user_item_features) + len(features)


@pytest.mark.core
@pytest.mark.parametrize(
    "features, is_query_features, is_item_features",
    [
        ([], False, True),
        ([], True, True),
        ([], True, False),
        (["timestamp"], True, True),
        (["gender"], True, True),
        (["category_id"], True, True),
    ],
)
@pytest.mark.usefixtures("full_pandas_dataset")
def test_label_encoder_transform(full_pandas_dataset, features, is_query_features, is_item_features):
    encoder = DatasetLabelEncoder()
    user_item_features = ["user_id", "item_id"]
    dataset = Dataset(
        feature_schema=get_features(full_pandas_dataset).subset(user_item_features + features),
        interactions=full_pandas_dataset["interactions"],
        query_features=full_pandas_dataset["users"] if is_query_features else None,
        item_features=full_pandas_dataset["items"] if is_item_features else None,
    )

    transformed_dataset = encoder.fit(dataset).transform(dataset)
    assert id(transformed_dataset) != id(dataset)
    assert transformed_dataset.interactions.equals(dataset.interactions[transformed_dataset.interactions.columns])

    if is_item_features:
        assert transformed_dataset.item_features.equals(
            dataset.item_features[transformed_dataset.item_features.columns]
        )
    if is_query_features:
        assert transformed_dataset.query_features.equals(
            dataset.query_features[transformed_dataset.query_features.columns]
        )


@pytest.mark.core
@pytest.mark.parametrize(
    "features, is_query_features, is_item_features",
    [
        ([], False, True),
        ([], True, True),
        ([], True, False),
        (["timestamp"], True, True),
        (["gender"], True, True),
        (["category_id"], True, True),
    ],
)
@pytest.mark.usefixtures("full_pandas_dataset")
def test_label_encoder_fit_transform(full_pandas_dataset, features, is_query_features, is_item_features):
    encoder = DatasetLabelEncoder()
    user_item_features = ["user_id", "item_id"]
    dataset = Dataset(
        feature_schema=get_features(full_pandas_dataset).subset(user_item_features + features),
        interactions=full_pandas_dataset["interactions"],
        query_features=full_pandas_dataset["users"] if is_query_features else None,
        item_features=full_pandas_dataset["items"] if is_item_features else None,
    )
    expected_transformed_dataset = encoder.fit(dataset).transform(dataset)
    actual_transformed_dataset = encoder.fit_transform(dataset)

    assert id(actual_transformed_dataset) != id(dataset)
    assert actual_transformed_dataset.interactions.equals(expected_transformed_dataset.interactions)


@pytest.mark.core
@pytest.mark.usefixtures("full_pandas_dataset")
def test_label_encoder_transform_unknown_feature(full_pandas_dataset):
    encoder = DatasetLabelEncoder()
    user_item_features = ["user_id", "item_id"]
    dataset_for_fit = Dataset(
        feature_schema=get_features(full_pandas_dataset).subset(user_item_features),
        interactions=full_pandas_dataset["interactions"],
    )
    encoder.fit(dataset_for_fit)

    dataset_for_transform = Dataset(
        feature_schema=get_features(full_pandas_dataset).subset([*user_item_features, "timestamp"]),
        interactions=full_pandas_dataset["interactions"],
    )

    encoder.transform(dataset_for_transform)
    key = "timestamp"

    assert key not in encoder._encoding_rules


@pytest.mark.core
@pytest.mark.usefixtures("full_pandas_dataset")
def test_not_initialized_encoding_rules(full_pandas_dataset):
    with pytest.raises(ValueError) as exc:
        DatasetLabelEncoder().transform(create_dataset(full_pandas_dataset))

    assert str(exc.value) == "Encoder is not initialized"


@pytest.mark.core
@pytest.mark.parametrize(
    "feature_source",
    [
        "user_id",
        "item_id",
        ["item_id", "user_id"],
    ],
)
@pytest.mark.usefixtures("full_pandas_dataset")
def test_get_encoder(full_pandas_dataset, feature_source):
    encoder = DatasetLabelEncoder()
    user_item_features = ["user_id", "item_id"]
    dataset_for_fit = Dataset(
        feature_schema=get_features(full_pandas_dataset).subset(user_item_features),
        interactions=full_pandas_dataset["interactions"],
    )
    encoder.fit(dataset_for_fit)
    encoder_from_get = encoder.get_encoder(feature_source)
    assert encoder_from_get is not None
    assert isinstance(encoder_from_get, LabelEncoder)


@pytest.mark.core
@pytest.mark.usefixtures("full_pandas_dataset")
def test_get_encoder_empty_rules_list(full_pandas_dataset):
    encoder = DatasetLabelEncoder()
    user_item_features = ["user_id", "item_id"]
    dataset_for_fit = Dataset(
        feature_schema=get_features(full_pandas_dataset).subset(user_item_features),
        interactions=full_pandas_dataset["interactions"],
    )
    encoder_from_get = encoder.fit(dataset_for_fit).get_encoder(["timestamp", "rating"])

    assert encoder_from_get is None


@pytest.mark.core
def test_label_encoder_properties(full_pandas_dataset):
    encoder = DatasetLabelEncoder()
    dataset = create_dataset(full_pandas_dataset)
    encoder.fit(dataset)

    assert isinstance(encoder.query_id_encoder, LabelEncoder)
    assert isinstance(encoder.item_id_encoder, LabelEncoder)
    assert isinstance(encoder.query_and_item_id_encoder, LabelEncoder)
    assert isinstance(encoder.query_features_encoder, LabelEncoder)
    assert isinstance(encoder.item_features_encoder, LabelEncoder)
    assert isinstance(encoder.interactions_encoder, LabelEncoder)
