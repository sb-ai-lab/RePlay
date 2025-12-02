from typing import Callable, Union
from typing_extensions import TypeAlias

import torch

GeneralValue: TypeAlias = Union[torch.Tensor, "GeneralBatch"]
GeneralBatch: TypeAlias = dict[str, GeneralValue]
GeneralCollateFn: TypeAlias = Callable[[GeneralBatch], GeneralBatch]
