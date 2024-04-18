import pytest

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.data.nn import TensorFeatureInfo, TensorFeatureSource
    from replay.data.nn.schema import TensorSchema
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder


@pytest.mark.torch
def test_tensor_feature_properties(some_num_tensor_feature, some_cat_tensor_feature):
    feature = TensorFeatureInfo("fake", feature_type=FeatureType.NUMERICAL)
    assert some_num_tensor_feature.feature_type == FeatureType.NUMERICAL
    assert some_cat_tensor_feature.feature_type == FeatureType.CATEGORICAL
    assert some_cat_tensor_feature.feature_hint in FeatureHint
    assert some_cat_tensor_feature.name == "cat_feature"
    assert some_num_tensor_feature.is_seq is True
    assert some_cat_tensor_feature.is_cat is True
    assert some_cat_tensor_feature.is_num is False
    assert some_num_tensor_feature.is_cat is False
    assert some_num_tensor_feature.is_num is True
    assert some_cat_tensor_feature.cardinality == some_cat_tensor_feature.embedding_dim
    assert some_num_tensor_feature.tensor_dim == 1
    assert isinstance(some_cat_tensor_feature.feature_source, TensorFeatureSource)
    assert len(some_num_tensor_feature.feature_sources) == 2
    assert len(some_cat_tensor_feature.feature_sources) == 1
    assert some_cat_tensor_feature.feature_source.index == 0
    assert feature.feature_source is None


@pytest.mark.torch
def test_tensor_scheme_properties(fake_schema, tensor_schema):
    schema = (
        TensorSchemaBuilder()
        .numerical(
            "item_id",
            tensor_dim=2,
            is_seq=True,
            feature_sources=[
                TensorFeatureSource(FeatureSource.INTERACTIONS, "item_feature1"),
                TensorFeatureSource(FeatureSource.INTERACTIONS, "item_feature2"),
            ],
            feature_hint=FeatureHint.RATING,
        )
        .build()
    )

    with pytest.raises(ValueError) as exc1:
        fake_schema.item()

    assert str(exc1.value) == "Only one element tensor schema can be converted to single feature"
    assert len(fake_schema) == 8
    assert fake_schema == fake_schema
    assert fake_schema != tensor_schema
    assert fake_schema.get("item_id").is_cat is True

    assert fake_schema.query_id_feature_name is None
    assert fake_schema.timestamp_feature_name is None
    assert tensor_schema.timestamp_feature_name is not None
    assert fake_schema.rating_feature_name is None
    assert fake_schema.item_id_feature_name is not None

    assert schema.all_features
    assert isinstance(schema.categorical_features, TensorSchema)
    assert isinstance(schema.numerical_features, TensorSchema)
    assert isinstance(schema.item_id_features, TensorSchema)
    assert isinstance(schema.query_id_features, TensorSchema)
    assert isinstance(schema.timestamp_features, TensorSchema)
    assert isinstance(schema.rating_features, TensorSchema)
    assert isinstance(schema.sequential_features, TensorSchema)
    assert schema.names == ["item_id"]
    assert schema.rating_feature_name == "item_id"


@pytest.mark.torch
def test_invalid_tensor_feature_type(some_num_tensor_feature):
    assert some_num_tensor_feature.feature_type in FeatureType


@pytest.mark.torch
def test_tensor_feature_setters(some_num_tensor_feature, some_cat_tensor_feature):
    some_num_tensor_feature._set_feature_hint(FeatureHint.RATING)
    some_num_tensor_feature._set_feature_sources(
        [
            TensorFeatureSource(FeatureSource.INTERACTIONS, "fake1"),
        ]
    )
    some_num_tensor_feature._set_tensor_dim(42)
    some_cat_tensor_feature._set_cardinality(42)
    some_cat_tensor_feature._set_embedding_dim(42)

    assert some_num_tensor_feature.feature_hint == FeatureHint.RATING
    assert len(some_cat_tensor_feature.feature_sources) == 1
    assert some_num_tensor_feature.tensor_dim == 42
    assert some_cat_tensor_feature.cardinality == 42
    assert some_cat_tensor_feature.embedding_dim == 42


@pytest.mark.torch
def test_tensor_feature_invalid_init():
    with pytest.raises(ValueError) as exc1:
        TensorFeatureInfo(name="fake", feature_type="unavailable type")

    with pytest.raises(ValueError) as exc2:
        TensorFeatureInfo(
            name="fake",
            feature_type=FeatureType.NUMERICAL,
            cardinality=42,
        )

    with pytest.raises(ValueError) as exc3:
        TensorFeatureInfo(
            name="fake",
            feature_type=FeatureType.CATEGORICAL,
            tensor_dim=42,
        )

    with pytest.raises(ValueError) as exc4:
        TensorFeatureInfo(
            name="fake",
            feature_type=FeatureType.NUMERICAL,
            feature_sources=[
                TensorFeatureSource(FeatureSource.INTERACTIONS, "fake1", 0),
                TensorFeatureSource(FeatureSource.INTERACTIONS, "fake2", 1),
            ],
        ).feature_source

    with pytest.raises(RuntimeError) as exc5:
        TensorFeatureInfo(
            name="fake",
            feature_type=FeatureType.NUMERICAL,
        ).cardinality

    with pytest.raises(RuntimeError) as exc6:
        TensorFeatureInfo(
            name="fake",
            feature_type=FeatureType.CATEGORICAL,
        ).tensor_dim

    with pytest.raises(RuntimeError) as exc7:
        TensorFeatureInfo(
            name="fake",
            feature_type=FeatureType.NUMERICAL,
        ).embedding_dim

    assert str(exc1.value) == "Unknown feature type"
    assert str(exc2.value) == "Cardinality and embedding dimensions are needed only with categorical feature type."
    assert str(exc3.value) == "Tensor dimensions is needed only with numerical feature type."
    assert str(exc4.value) == "Only one element feature sources can be converted to single feature source."
    assert str(exc5.value) == "Can not get cardinality because feature type of fake column is not categorical."
    assert str(exc6.value) == "Can not get tensor dimensions because feature type of fake feature is not numerical."
    assert str(exc7.value) == (
        "Can not get embedding dimensions because" " feature type of fake feature is not categorical."
    )


@pytest.mark.torch
def test_tensor_scheme_inits():
    features_list = [
        TensorFeatureInfo(
            "item_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
            cardinality=6,
            feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
        ),
        TensorFeatureInfo(
            "user_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.QUERY_ID,
            cardinality=6,
            feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id")],
        ),
    ]

    feature = TensorFeatureInfo(
        "rating",
        feature_type=FeatureType.NUMERICAL,
        tensor_dim=6,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "rating")],
    )

    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            "user_id",
            cardinality=6,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id"),
            feature_hint=FeatureHint.QUERY_ID,
        )
        .numerical("rating", tensor_dim=6, feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "rating")])
        .build()
    )

    assert TensorSchema(features_list) is not None
    assert TensorSchema(feature) is not None
    assert TensorSchema([*features_list, feature]).names == schema.names
    assert (TensorSchema(features_list) + TensorSchema(feature)).names == schema.names
