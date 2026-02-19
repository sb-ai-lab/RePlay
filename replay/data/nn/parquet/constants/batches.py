from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch

GeneralValue: TypeAlias = "torch.Tensor" | "GeneralBatch"
GeneralBatch: TypeAlias = dict[str, GeneralValue]
GeneralCollateFn: TypeAlias = Callable[[GeneralBatch], GeneralBatch]
