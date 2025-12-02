import pytest
import torch

from replay.data.parquet.impl.indexing import get_mask, get_offsets


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("length", [1, 35, 42, 1023])
@pytest.mark.parametrize("max_seq_len", [1, 2, 17, 123])
def test_get_offsets(seed: int, length: int, max_seq_len: int):
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(
        low=0,
        high=max_seq_len,
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )

    offsets: torch.LongTensor = get_offsets(lengths)
    assert torch.numel(offsets) == length + 1

    test_lengths: torch.LongTensor = offsets[1:] - offsets[:-1]
    assert torch.all(test_lengths == lengths).cpu().item()


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("run_count", [1, 2, 3, 7])
@pytest.mark.parametrize("length", [1, 35, 42, 1023])
@pytest.mark.parametrize("max_seq_len", [1, 2, 17, 123])
def test_get_mask(seed: int, run_count: int, length: int, max_seq_len: int):
    gen = torch.Generator().manual_seed(seed)
    lengths= torch.randint(
        low=0,
        high=int(1.5 * max_seq_len),
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )
    offsets = get_offsets(lengths)
    run_lengths = torch.randint(
        low=1,
        high=2 * length,
        size=(run_count,),
        generator=gen,
        dtype=torch.int64,
    )

    for run in range(run_count):
        indices_count: int = run_lengths[run].cpu().item()
        indices: torch.LongTensor = torch.randint(
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

        true_count: torch.LongTensor = torch.minimum(len_ids, torch.asarray(max_seq_len))
        assert torch.all(torch.sum(mask, dtype=torch.int64, dim=-1) == true_count).cpu().item()

        true_last: torch.LongTensor = torch.where(zero_len, 0, (last_ids - 1))
        assert torch.all(torch.max(output, dim=-1).values == true_last).cpu().item()
