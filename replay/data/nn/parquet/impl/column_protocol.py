from typing import Protocol

import torch

OutputType = tuple[torch.BoolTensor, torch.Tensor]


class ColumnProtocol(Protocol):
    def __len__(self) -> int: ...

    @property
    def length(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def __getitem__(self, indices: torch.LongTensor) -> OutputType: ...
