import pytest
import torch

from replay.data import FeatureType
from replay.data.nn.schema import TensorFeatureInfo, TensorSchema
from replay.nn import SequenceEmbedding, CategoricalEmbedding


@pytest.mark.parametrize("excluded_features", [None, ["num_feature"], ["item_id", "num_feature"]])
def test_sequence_embedding_forward(tensor_schema, simple_batch, excluded_features):
    embedder = SequenceEmbedding(tensor_schema, excluded_features=excluded_features)
    output_tensors = embedder(simple_batch["feature_tensors"])

    for feature in simple_batch["feature_tensors"].keys():
        if excluded_features is not None and feature in excluded_features:
            assert feature not in output_tensors.keys()
        else:
            assert output_tensors[feature].size() == (4, 5, 64)


def test_sequence_embedding_get_embedding_dim(tensor_schema):
    embedder = SequenceEmbedding(tensor_schema)
    for feature in tensor_schema.keys():
        assert embedder.embeddings_dim[feature] == 64


@pytest.mark.parametrize(
    "indices",
    [
        torch.LongTensor([1]),
        torch.LongTensor([0, 1, 2]),
        torch.LongTensor([[0, 1, 2], [1, 1, 1]]),
        torch.LongTensor([[0, 2], [1, 1], [3, 2]]),
        None,
    ],
)
def test_sequence_embedding_get_item_weights(tensor_schema, indices):
    embedder = SequenceEmbedding(tensor_schema)
    feature_cardinality_wo_pad = tensor_schema["item_id"].cardinality - 1
    expected_shape = indices.shape if indices is not None else (feature_cardinality_wo_pad,)

    assert embedder.get_item_weights(indices).size() == (*expected_shape, 64)


def test_sequence_embedding_get_features_weights(tensor_schema):
    embedder = SequenceEmbedding(tensor_schema)

    for feature in tensor_schema.categorical_features.keys():
        emb = embedder.feature_embedders[feature].weight
        feature_cardinality_wo_pad = tensor_schema[feature].cardinality - 1
        assert emb.shape == (feature_cardinality_wo_pad, 64)

    for feature in tensor_schema.numerical_features.keys():
        emb = embedder.feature_embedders[feature].weight
        assert emb.shape == (64, tensor_schema[feature].tensor_dim)


def test_wrong_feature_type():
    tensor_schema = TensorSchema(
        [
            TensorFeatureInfo(name="some_feature", is_seq=False, feature_type=FeatureType.CATEGORICAL),
        ]
    )
    with pytest.raises(NotImplementedError):
        SequenceEmbedding(tensor_schema)


def test_exclude_all_keys(tensor_schema):
    excluded_features = tensor_schema.keys()
    with pytest.raises(ValueError):
        SequenceEmbedding(tensor_schema, excluded_features=excluded_features)

def test_warnings_categorical_emb():
    tensor_info = TensorFeatureInfo(
        name="some_feature", 
        is_seq=True,
        embedding_dim=64,
        feature_type=FeatureType.CATEGORICAL,
        cardinality=5,
        padding_value=2
    )
    with pytest.warns(UserWarning):
        embedding = CategoricalEmbedding(tensor_info)
        assert embedding.weight.size() == (4, 64)
