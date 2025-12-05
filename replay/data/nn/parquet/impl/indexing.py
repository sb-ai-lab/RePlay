from typing import Union

import torch


def raw_get_offsets(lengths: torch.LongTensor) -> torch.LongTensor:
    zero = torch.zeros((1,), device=lengths.device, dtype=torch.int64)
    cumsum = torch.cumsum(lengths, dim=-1)
    return torch.cat([zero, cumsum])


def get_offsets(lengths: torch.LongTensor) -> torch.LongTensor:
    if lengths.ndim != 1:
        msg = f"Lengths must be strictly 1D. Got {lengths.ndim}D."
        raise ValueError(msg)
    min_length = torch.min(lengths.detach()).cpu().item()
    if min_length < 0:
        msg = f"There is a negative length. Got {min_length}."
        raise ValueError(msg)
    return raw_get_offsets(lengths)


LengthType = Union[int, torch.LongTensor]


def raw_get_mask(
    indices: torch.LongTensor,
    offsets: torch.LongTensor,
    length: LengthType,
) -> tuple[torch.BoolTensor, torch.LongTensor]:
    """
    Performs mask construction.

    For
    """
    length = torch.asarray(length, dtype=torch.int64, device=indices.device)

    # For every "line", start element index matches the offset, while end is the offset of the next line
    last = offsets[indices + 1]
    first = offsets[indices + 0]

    per_line = length - (last - first)

    arange = torch.arange(length, dtype=torch.int64, device=offsets.device)
    raw_indices = (first[:, None] - per_line[:, None]) + arange[None, :]
    mask = (first[:, None] <= raw_indices) & (raw_indices < last[:, None])

    assert torch.all(torch.sum(mask, dim=-1, dtype=torch.int64) == torch.minimum(last - first, length)).cpu().item()
    # We are indexing `first` (not anymore lmao) because of the data locality & tests
    output_indices = torch.where(mask, raw_indices, 0)
    assert torch.all((torch.max(output_indices, dim=-1).values < last) | (last == first)).cpu().item()
    return (mask, output_indices)


def get_mask(
    indices: torch.LongTensor,
    offsets: torch.LongTensor,
    length: LengthType,
) -> tuple[torch.BoolTensor, torch.LongTensor]:
    """
    Perform input sanity checks, then contructs a mask from inputs.

    The mask calculation itself is performed via the ``raw_get_mask`` method.

    Args:
        indices (torch.LongTensor): _description_
        offsets (torch.LongTensor): _description_
        length (LengthType): _description_

    Raises:
        ValueError: _description_
        IndexError: _description_
        RuntimeError: _description_
        ValueError: _description_
        IndexError: _description_
        IndexError: _description_
        ValueError: _description_

    Returns:
        tuple[torch.BoolTensor, torch.LongTensor]: _description_
    """
    if torch.asarray(length).cpu().item() < 1:
        msg = f"Length must be a positive number. Got {length}"
        raise ValueError(msg)
    if torch.numel(indices) < 1:
        msg = f"Indices must be non-empty. Got {torch.numel(indices)}."
        raise IndexError(msg)
    if indices.device != offsets.device:  # pragma: no cover
        msg = f"Devices must match. Got {indices.device} vs {offsets.device}"
        raise RuntimeError(msg)
    if offsets.ndim != 1:
        msg = f"Offsets must be strictly 1D. Got {offsets.ndim}D."
        raise ValueError(msg)
    min_index = torch.min(indices.detach()).cpu().item()
    if min_index < 0:
        msg = f"Index is too small. Got {min_index}."
        raise IndexError(msg)
    max_index = torch.max(indices.detach()).cpu().item()
    if torch.numel(offsets) < max_index:
        msg = f"Index is too large. Got {max_index}."
        raise IndexError(msg)
    if not torch.all(offsets[:-1] <= offsets[1:]).cpu().item():
        msg = "Offset sequence is not monothonous."
        raise ValueError(msg)
    return raw_get_mask(indices, offsets, length)
