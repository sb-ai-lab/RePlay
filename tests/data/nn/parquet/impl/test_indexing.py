import pytest
import torch
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from replay.data.nn.parquet.impl.indexing import get_mask, get_offsets

settings.load_profile("fast")


@given(
    seed=st.integers(min_value=0, max_value=torch.iinfo(torch.int64).max),
    length=st.integers(min_value=1, max_value=1024),
    max_seq_len=st.integers(min_value=1, max_value=128),
)
def test_get_offsets(seed: int, length: int, max_seq_len: int):
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(
        low=0,
        high=max_seq_len,
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )

    offsets = get_offsets(lengths)
    assert torch.numel(offsets) == length + 1

    test_lengths = offsets[1:] - offsets[:-1]
    assert torch.all(test_lengths == lengths).cpu().item()


def test_invalid_offsets():
    with pytest.raises(ValueError) as exc:
        get_offsets(torch.tensor([[0, 1], [1, 0]]))
    assert "Lengths must be strictly 1D." in str(exc.value)

    with pytest.raises(ValueError) as exc:
        get_offsets(torch.tensor([0, 1, -1]))
    assert "There is a negative length." in str(exc.value)


@given(
    seed=st.integers(min_value=0, max_value=torch.iinfo(torch.int64).max),
    length=st.integers(min_value=1, max_value=1024),
    max_seq_len=st.integers(min_value=1, max_value=128),
)
def test_get_mask(seed: int, length: int, max_seq_len: int):
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(
        low=0,
        high=int(1.5 * max_seq_len),
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )
    offsets = get_offsets(lengths)
    indices_count = (
        torch.randint(
            low=1,
            high=2 * length,
            size=(1,),
            generator=gen,
            dtype=torch.int64,
        )
        .cpu()
        .item()
    )

    indices = torch.randint(
        low=0,
        high=length,
        size=(indices_count,),
        generator=gen,
        dtype=torch.int64,
    )

    mask, output = get_mask(indices, offsets, max_seq_len)

    assert mask.size() == (indices_count, max_seq_len)

    first_ids = offsets[:-1][indices]
    last_ids = offsets[1:][indices]
    len_ids = last_ids - first_ids
    zero_len = len_ids == 0

    true_count = torch.minimum(len_ids, torch.asarray(max_seq_len))
    assert torch.all(torch.sum(mask, dtype=torch.int64, dim=-1) == true_count).cpu().item()

    true_last = torch.where(zero_len, 0, (last_ids - 1))
    assert torch.all(torch.max(output, dim=-1).values == true_last).cpu().item()


@pytest.mark.parametrize(
    "indices, offsets, length, expected_error, expected_log",
    [
        (None, None, 0, ValueError, "Length must be a positive number."),
        (torch.tensor([]), None, 1, IndexError, "Indices must be non-empty."),
        (torch.tensor([1]), torch.tensor([[1]]), 1, ValueError, "Offsets must be strictly 1D."),
        (torch.tensor([-1]), torch.tensor([1]), 1, IndexError, "Index is too small."),
        (torch.tensor([3]), torch.tensor([1, 2]), 1, IndexError, "Index is too large."),
        (torch.tensor([1]), torch.tensor([1, 0]), 1, ValueError, "Offset sequence is not monothonous."),
    ],
)
def test_invalid_masks(indices, offsets, length, expected_error, expected_log):
    with pytest.raises(expected_error) as exc:
        get_mask(indices, offsets, length)
    assert expected_log in str(exc.value)


@pytest.mark.skip("Driver-dependent. Needs rework to decouple from CUDA.")
def test_mismatching_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", True)

    with pytest.raises(RuntimeError) as exc:
        get_mask(torch.tensor([1]), torch.tensor([1], device="cuda"), 1)
    assert "Devices must match." in str(exc.value) or "Found no NVIDIA driver on your system." in str(exc.value)
