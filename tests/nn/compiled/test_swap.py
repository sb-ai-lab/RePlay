import pytest

pytest.importorskip("torch")
import torch

from replay.data import FeatureType
from replay.data.nn import TensorFeatureInfo
from replay.nn.compiled._swap import ExportableEmbeddingBag, swap_embedding_bags
from replay.nn.embedding import CategoricalEmbedding

CARDINALITY, EMB_DIM, ROWS, TOKENS = 8, 16, 5, 6
PADDING = CARDINALITY


def _indices():
    torch.manual_seed(11)
    indices = torch.randint(0, CARDINALITY + 1, (ROWS, TOKENS))
    indices[1] = PADDING
    indices[2, ::2] = PADDING
    return indices


def _bag(mode):
    torch.manual_seed(7)
    return torch.nn.EmbeddingBag(CARDINALITY + 1, EMB_DIM, padding_idx=PADDING, mode=mode)


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_bag_equivalence(mode):
    bag = _bag(mode)
    swapped = ExportableEmbeddingBag(bag)
    bag.eval()
    swapped.eval()
    with torch.no_grad():
        original = bag(_indices())
        replaced = swapped(_indices())
    tolerance = {} if mode == "max" else {"atol": 1e-5, "rtol": 1e-5}
    assert torch.allclose(original, replaced, **tolerance)
    assert torch.equal(original[1], replaced[1])
    assert torch.all(original[1] == 0)


def test_weights_are_tied():
    bag = _bag("sum")
    swapped = ExportableEmbeddingBag(bag)
    assert swapped.emb.weight is bag.weight
    assert swapped.weight is bag.weight
    assert swapped.num_embeddings == bag.num_embeddings
    assert swapped.embedding_dim == bag.embedding_dim
    assert swapped.padding_idx == bag.padding_idx
    assert swapped.mode == "sum"


def test_fill_padding_idx_with_zero():
    bag = _bag("mean")
    swapped = ExportableEmbeddingBag(bag)
    with torch.no_grad():
        swapped.emb.weight[PADDING].fill_(1.0)
    swapped._fill_padding_idx_with_zero()
    assert torch.all(swapped.weight[PADDING] == 0)


def test_swap_recursive_through_categorical_embedding():
    info = TensorFeatureInfo(
        "genres",
        is_seq=True,
        feature_type=FeatureType.CATEGORICAL_LIST,
        embedding_dim=EMB_DIM,
        cardinality=CARDINALITY,
        padding_value=PADDING,
    )
    embedding = CategoricalEmbedding(info)
    embedding.eval()
    indices = _indices().unsqueeze(1).expand(ROWS, 4, TOKENS).contiguous()
    with torch.no_grad():
        before = embedding(indices)
    assert swap_embedding_bags(embedding) == 1
    with torch.no_grad():
        after = embedding(indices)
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
