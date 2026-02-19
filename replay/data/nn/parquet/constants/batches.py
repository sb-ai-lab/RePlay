from collections.abc import Callable
from typing import TypeAlias, Union

import torch

# Union is here specifically to stay until Python 3.12+
GeneralValue: TypeAlias = Union[torch.Tensor, "GeneralBatch"]
GeneralBatch: TypeAlias = dict[str, GeneralValue]
GeneralCollateFn: TypeAlias = Callable[[GeneralBatch], GeneralBatch]
