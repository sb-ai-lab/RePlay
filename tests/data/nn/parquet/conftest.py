import pytest

pytest.importorskip("torch", reason="Module 'torch' is required for ParquetDataset tests.")

import itertools
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from replay.constants.metadata import PADDING_FLAG, SHAPE_FLAG
from replay.data.nn.parquet.impl.indexing import get_offsets
from replay.data.nn.parquet.metadata import Metadata
from replay.data.utils.typing.dtype import torch_to_pyarrow

PADDING: int = -2
MAX_PHASE: float = 2.0 * torch.pi

X: str = "x"
SIN_X: str = "sin_x"
COS_X: str = "cos_x"
SIN_SIN: str = "sin_sin"
SIN_COS: str = "sin_cos"
COS_COS: str = "cos_cos"
COS_SIN: str = "cos_sin"

PHASE: str = "phase"
OFFSET: str = "offset"
FREQUENCY: str = "frequency"

LENGTH: str = "length"
SIN_LENGTH: str = "sin_length"
COS_LENGTH: str = "cos_length"

INDEX: str = "index"
SIN_INDEX: str = "sin_index"
COS_INDEX: str = "cos_index"

Batch = Dict[str, torch.Tensor]


def get_absolute_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    count: int = offsets[-1].detach().cpu().item()
    return torch.arange(count, device=offsets.device, dtype=torch.int64)


def get_rows(offsets: torch.LongTensor) -> torch.LongTensor:
    indices: torch.LongTensor = get_absolute_indices(offsets)
    rows: torch.LongTensor = torch.searchsorted(offsets, indices, side="right")
    return torch.clamp(rows, max=torch.numel(offsets) - 1) - 1


def get_relative_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    abs_indices: torch.LongTensor = get_absolute_indices(offsets)
    rows: torch.LongTensor = get_rows(offsets)
    return abs_indices - offsets[rows]


def get_mask(offsets: torch.LongTensor, length: int) -> Tuple[torch.BoolTensor, torch.LongTensor]:
    last: torch.LongTensor = offsets[1:]
    first: torch.LongTensor = offsets[:-1]
    per_line: torch.LongTensor = length - (last - first)
    arange: torch.Longtensor = torch.arange(length, dtype=torch.int64, device=offsets.device)
    raw_indices: torch.LongTensor = (first[:, None] - per_line[:, None]) + arange[None, :]
    mask: torch.BoolTensor = (first[:, None] <= raw_indices) & (raw_indices < last[:, None])
    indices: torch.LongTensor = torch.where(mask, raw_indices, 0)
    return (mask, indices)


def get_padded(offsets: torch.LongTensor, series: torch.Tensor, length: int) -> Tuple[torch.BoolTensor, torch.Tensor]:
    mask, indices = get_mask(offsets, length)
    gathered: torch.Tensor = torch.take(series, indices)
    padded: torch.Tensor = torch.where(mask, gathered, PADDING)
    return (mask, padded)


@dataclass
class BatchGenerator:
    min_length: int = 1
    max_length: int = 512
    mean_length: int = 128

    min_frequency: float = 0.0
    max_frequency: float = 20.0
    mean_frequency: float = 5.0

    batch_size: int = 4_096

    generator: Optional[torch.Generator] = None

    def generate_lengths(self) -> torch.LongTensor:
        ones: torch.Tensor = torch.ones(self.batch_size, dtype=torch.float32)
        raw: torch.Tensor = torch.poisson(ones * self.mean_length, generator=self.generator)
        return torch.clip(raw.to(dtype=torch.int64), min=self.min_length, max=self.max_length)

    def generate_phases(self) -> torch.FloatTensor:
        return MAX_PHASE * torch.rand(self.batch_size, dtype=torch.float32, generator=self.generator)

    def generate_frequencies(self) -> torch.FloatTensor:
        ones: torch.Tensor = torch.ones(self.batch_size, dtype=torch.float32)
        raw: torch.Tensor = torch.poisson(ones * self.mean_frequency, generator=self.generator)
        noise: torch.Tensor = torch.rand(self.batch_size, dtype=torch.float32, generator=self.generator)
        return MAX_PHASE * torch.clip((raw + noise), min=self.min_frequency, max=self.max_frequency)

    def generate_xs(self) -> Batch:
        lengths: torch.Tensor = self.generate_lengths()
        offsets: torch.Tensor = get_offsets(lengths)
        rel_indices: torch.Tensor = get_relative_indices(offsets)
        unique_xs: torch.Tensor = torch.linspace(0, 1, self.max_length, dtype=torch.float32, device=rel_indices.device)
        xs: torch.Tensor = unique_xs[rel_indices]
        return {
            X: xs,
            LENGTH: lengths,
            INDEX: rel_indices,
        }

    def generate_base(self) -> Batch:
        cos_xs: Batch = self.generate_xs()
        sin_xs: Batch = self.generate_xs()
        phases: torch.FloatTensor = self.generate_phases()
        frequencies: torch.FloatTensor = self.generate_frequencies()
        return {
            PHASE: phases,
            FREQUENCY: frequencies,
            SIN_LENGTH: sin_xs[LENGTH],
            COS_LENGTH: cos_xs[LENGTH],
            SIN_X: sin_xs[X],
            COS_X: cos_xs[X],
            SIN_INDEX: sin_xs[INDEX],
            COS_INDEX: cos_xs[INDEX],
        }

    def sin(self, base: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        xs: torch.Tensor = base[SIN_X]
        phase: torch.Tensor = base[PHASE]
        frequency: torch.Tensor = base[FREQUENCY]
        lengths: torch.Tensor = base[SIN_LENGTH]
        offsets: torch.Tensor = get_offsets(lengths)
        rows: torch.Tensor = get_rows(offsets)
        phases: torch.Tensor = phase[rows]
        frequencies: torch.Tensor = frequency[rows]
        cos: torch.Tensor = torch.cos(phases + frequencies * xs)
        sin: torch.Tensor = torch.sin(phases + frequencies * xs)
        return (cos, sin)

    def cos(self, base: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        xs: torch.Tensor = base[COS_X]
        phase: torch.Tensor = base[PHASE]
        frequency: torch.Tensor = base[FREQUENCY]
        lengths: torch.Tensor = base[COS_LENGTH]
        offsets: torch.Tensor = get_offsets(lengths)
        rows: torch.Tensor = get_rows(offsets)
        phases: torch.Tensor = phase[rows]
        frequencies: torch.Tensor = frequency[rows]
        cos: torch.Tensor = torch.cos(phases + frequencies * xs)
        sin: torch.Tensor = torch.sin(phases + frequencies * xs)
        return (cos, sin)

    def generate_batch(self) -> Batch:
        base: Batch = self.generate_base()
        sin_cos, sin_sin = self.sin(base)
        cos_cos, cos_sin = self.cos(base)
        return {
            COS_COS: cos_cos,
            COS_SIN: cos_sin,
            SIN_SIN: sin_sin,
            SIN_COS: sin_cos,
            **base,
        }

    def sin_padded(self, batch: Batch) -> Batch:
        length: torch.Tensor = batch[SIN_LENGTH]
        offsets: torch.Tensor = get_offsets(length)
        _, xs = get_padded(offsets, batch[SIN_X], self.max_length)
        _, cos = get_padded(offsets, batch[SIN_COS], self.max_length)
        _, sin = get_padded(offsets, batch[SIN_SIN], self.max_length)
        _, idx = get_padded(offsets, batch[SIN_INDEX], self.max_length)
        return {SIN_X: xs, SIN_COS: cos, SIN_SIN: sin, SIN_INDEX: idx}

    def cos_padded(self, batch: Batch) -> Batch:
        length: torch.Tensor = batch[COS_LENGTH]
        offsets: torch.Tensor = get_offsets(length)
        _, xs = get_padded(offsets, batch[COS_X], self.max_length)
        _, cos = get_padded(offsets, batch[COS_COS], self.max_length)
        _, sin = get_padded(offsets, batch[COS_SIN], self.max_length)
        _, idx = get_padded(offsets, batch[COS_INDEX], self.max_length)
        return {COS_X: xs, COS_COS: cos, COS_SIN: sin, COS_INDEX: idx}

    def to_padded(self, batch: Batch) -> Batch:
        cos: Batch = self.cos_padded(batch)
        sin: Batch = self.sin_padded(batch)

        def norm(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[:, None]

        return {
            **cos,
            **sin,
            PHASE: norm(batch[PHASE]),
            FREQUENCY: norm(batch[FREQUENCY]),
            COS_LENGTH: norm(batch[COS_LENGTH]),
            SIN_LENGTH: norm(batch[SIN_LENGTH]),
        }

    def generate_padded(self) -> Batch:
        base: Batch = self.generate_batch()
        return self.to_padded(base)


def to_array(vals: torch.Tensor) -> pa.Array:
    values: list = vals.ravel().cpu().tolist()
    return pa.array(values, type=torch_to_pyarrow(vals.dtype))


def to_list_array(lens: torch.LongTensor, vals: torch.Tensor) -> pa.Array:
    values: list = vals.cpu().tolist()
    dtype: pa.DataType = torch_to_pyarrow(vals.dtype)
    offsets: torch.LongTensor = get_offsets(lens).cpu().tolist()
    raw_list: list = [values[i[0] : i[1]] for i in itertools.pairwise(offsets)]
    return pa.array(raw_list, type=pa.list_(dtype))


def make_metadata(batch_generator: BatchGenerator) -> Metadata:
    return {
        PHASE: {},
        FREQUENCY: {},
        SIN_LENGTH: {},
        COS_LENGTH: {},
        SIN_X: {
            PADDING_FLAG: PADDING,
            SHAPE_FLAG: batch_generator.max_length,
        },
        SIN_SIN: {
            PADDING_FLAG: PADDING,
            SHAPE_FLAG: batch_generator.max_length,
        },
        COS_X: {
            PADDING_FLAG: PADDING,
            SHAPE_FLAG: batch_generator.max_length,
        },
        COS_COS: {
            PADDING_FLAG: PADDING,
            SHAPE_FLAG: batch_generator.max_length,
        },
    }


def batch_to_pyarrow(batch: Batch) -> pa.Table:
    phase: pa.Array = to_array(batch[PHASE])
    freq: pa.Array = to_array(batch[FREQUENCY])
    sin_len: pa.Array = to_array(batch[SIN_LENGTH])
    cos_len: pa.Array = to_array(batch[COS_LENGTH])
    sin_x: pa.Array = to_list_array(batch[SIN_LENGTH], batch[SIN_X])
    cos_x: pa.Array = to_list_array(batch[COS_LENGTH], batch[COS_X])
    sin_sin: pa.Array = to_list_array(batch[SIN_LENGTH], batch[SIN_SIN])
    cos_cos: pa.Array = to_list_array(batch[COS_LENGTH], batch[COS_COS])
    return pa.table(
        {
            PHASE: phase,
            FREQUENCY: freq,
            SIN_LENGTH: sin_len,
            COS_LENGTH: cos_len,
            SIN_SIN: sin_sin,
            COS_COS: cos_cos,
            SIN_X: sin_x,
            COS_X: cos_x,
        },
    )


def write_dataset(batch_generator: BatchGenerator, path: str, batch_count: int, write_every_n_batch: int = 8) -> str:
    writer: Optional[pq.ParquetWriter] = None

    batch_idx: int
    for batch_idx in range(batch_count):
        raw_batch: Batch = batch_generator.generate_batch()
        batch_table: pa.Table = batch_to_pyarrow(raw_batch)

        if batch_idx % write_every_n_batch == 0:
            if writer is not None:
                writer.close()
                writer = None
            filename: str = f"part_{batch_idx:06d}.parquet"
            file_path: str = os.path.join(path, filename)

            assert writer is None
            writer = pq.ParquetWriter(file_path, batch_table.schema)

        assert writer is not None

        writer.write(batch_table)

    if writer is not None:
        writer.close()
        writer = None

    return path
