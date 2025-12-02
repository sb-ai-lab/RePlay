import pytest
import torch

from replay.data.parquet.impl.indexing import get_mask, get_offsets
from replay.data.parquet.impl.array_1d_column import Array1DColumn


@pytest.mark.parametrize("seed", [1, 19, 777])
@pytest.mark.parametrize("run_count", [1, 2, 3, 5])
@pytest.mark.parametrize("length", [0, 1, 17, 33, 123])
@pytest.mark.parametrize("seq_len", [1, 2, 7, 77, 128])
def test_sequence(seed: int, run_count: int, length: int, seq_len: int):
    gen = torch.Generator().manual_seed(seed)

    if length != 0:
        lengths = torch.randint(
            low=1,
            high=int(1.5 * seq_len) + 1,
            size=(length,),
            generator=gen,
            dtype=torch.int64,
        )
        offsets = get_offsets(lengths)
        data = torch.arange(offsets[-1].cpu().item(), dtype=torch.int64) * seed
    else:
        lengths = torch.tensor([])
        offsets = torch.tensor([])
        data = torch.tensor([])

    sequence = Array1DColumn(data, lengths, seq_len, padding=-seed)

    run_lengths = torch.randint(
        low=1,
        high=2 * length,
        size=(run_count,),
        generator=gen,
        dtype=torch.int64,
    )

    for run in range(run_count):
        indices_count = run_lengths[run].cpu().item()
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
