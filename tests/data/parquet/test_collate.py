from typing import Dict

import torch

from replay.data.parquet.collate import dict_collate

Batch = Dict[str, torch.Tensor]


def test_dict_collate():
    batch_size: int = 5

    def make_sample(index: int) -> Batch:
        sample: Batch = {
            "x": torch.asarray([index], dtype=torch.int64).reshape(1, -1),
            "y": torch.asarray([index, index], dtype=torch.int64).reshape(1, -1),
            "z": torch.asarray([index, index, index], dtype=torch.int64).reshape(1, -1),
        }
        return sample

    samples: Batch = [make_sample(index) for index in range(batch_size)]
    batch: Batch = dict_collate(samples)

    gtr: torch.Tensor = torch.arange(batch_size, dtype=torch.int64)

    assert batch["x"].shape == (batch_size, 1)
    assert torch.all(batch["x"] == gtr[:, None]).cpu().item()

    assert batch["y"].shape == (batch_size, 2)
    assert torch.all(batch["y"] == gtr[:, None]).cpu().item()

    assert batch["z"].shape == (batch_size, 3)
    assert torch.all(batch["z"] == gtr[:, None]).cpu().item()
