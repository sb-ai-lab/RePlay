import pytest
import torch

from replay.nn import ConcatAggregator


@pytest.mark.parametrize("tensor_shapes", [[64, 128, 64], [64, 64, 64, 64, 64, 64, 64], [1, 3, 32], [64], [128, 1]])
def test_agg_concat_forward(tensor_shapes):
    aggregator = ConcatAggregator(input_embedding_dims=tensor_shapes, output_embedding_dim=64)

    feature_tensors = {f"dummy_tensor_{i}": torch.rand(25, shape) for i, shape in enumerate(tensor_shapes)}

    aggregated_hidden = aggregator(feature_tensors)
    assert aggregated_hidden.shape == (25, 64)


def test_agg_concat_forward_with_size_mismatch():
    with pytest.raises(ValueError):
        ConcatAggregator(input_embedding_dims=[32], output_embedding_dim=64)
