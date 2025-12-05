from typing import Callable, Union

import torch
from typing_extensions import TypeAlias

GeneralValue: TypeAlias = Union[torch.Tensor, "GeneralBatch"]
GeneralBatch: TypeAlias = dict[str, GeneralValue]
GeneralCollateFn: TypeAlias = Callable[[GeneralBatch], GeneralBatch]
