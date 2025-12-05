import pytest
import torch
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from replay.data.nn.parquet.impl.array_1d_column import Array1DColumn
from replay.data.nn.parquet.impl.indexing import get_mask, get_offsets

settings.load_profile("fast")


@given(
    seed=st.integers(min_value=0, max_value=torch.iinfo(torch.int64).max),
    length=st.integers(min_value=1, max_value=1024),
    seq_len=st.integers(min_value=1, max_value=128),
)
def test_array1d(seed: int, length: int, seq_len: int):
    gen = torch.Generator().manual_seed(seed)

    lengths = torch.randint(
        low=0,
        high=int(1.5 * seq_len) + 1,
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )
    offsets = get_offsets(lengths)
    data = torch.arange(offsets[-1].cpu().item(), dtype=torch.int64) * seed

    sequence = Array1DColumn(data, lengths, seq_len, padding=-seed)
    assert len(sequence) == length

    indices_count = torch.randint(
        low=1,
        high=2 * length,
        size=(1,),
        generator=gen,
        dtype=torch.int64,
    )

    indices = torch.randint(
        low=0,
        high=length,
        size=(indices_count,),
        generator=gen,
        dtype=torch.int64,
    )

    mask, output = sequence[indices]
    gtr_mask, gtr_ids = get_mask(indices, offsets, seq_len)

    assert torch.all(gtr_mask == mask).cpu().all()

    gtr_vals = torch.where(gtr_mask, gtr_ids * seed, -seed)

    assert torch.all(gtr_vals == output).cpu().item()


def test_invalid_shape():
    INVALID_LENGTHS_SHAPE = [1, 2]

    lengths = torch.randint(
        low=0,
        high=int(1.5 * 5) + 1,
        size=INVALID_LENGTHS_SHAPE,
        dtype=torch.int64,
    )
    data = torch.arange(42, dtype=torch.int64)

    with pytest.raises(ValueError) as exc:
        _ = Array1DColumn(data, lengths, INVALID_LENGTHS_SHAPE, padding=-1)
    assert "Array1DColumn accepts a shape of size (1,)" in str(exc.value)
