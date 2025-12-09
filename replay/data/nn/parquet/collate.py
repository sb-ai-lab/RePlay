from collections.abc import Sequence

import torch

from replay.data.nn.parquet.constants.batches import GeneralBatch, GeneralValue


def dict_collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Simple collate function that converts a dict of values into a tensor dict."""
    return {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0]}


def general_collate(batch: Sequence[GeneralBatch]) -> GeneralBatch:
    """General collate function that converts a nested dict of values into a tensor dict."""
    result = {}
    test_sample = batch[0]

    if len(batch) == 1:
        return test_sample

    for key, test_value in test_sample.items():
        values: Sequence[GeneralValue] = [sample[key] for sample in batch]
        if torch.is_tensor(test_value):
            result[key] = torch.cat(values, dim=0)
        else:
            assert isinstance(test_value, dict)
            result[key] = general_collate(values)

    return result
