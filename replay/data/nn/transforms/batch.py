import inspect
from typing import Generic, NamedTuple, TypeVar, Union

import torch
from typing_extensions import Self

from replay.data.nn.transforms.base import BaseTransform

T = TypeVar("T", bound=NamedTuple)


def is_namedtuple(obj):
    return inspect.isclass(obj) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


class BatchingTransform(BaseTransform, Generic[T]):
    """
    Transforms a batch dict into an specified NamedTuple of features.
    For example, `SasRecTrainingBatch` may be a target NamedTuple for converting source batch.
    """

    def __init__(self: Self, target: Union[T, tuple[str]]) -> None:
        """
        :param target: NamedTuple class into which source batch will be converted.
        """
        super().__init__()
        self.target = target

    def forward(self: Self, batch: dict[str, torch.Tensor]) -> Union[T, tuple[torch.Tensor]]:
        batch_fields = set(self.target._fields)

        if not batch_fields.issubset(set(batch.keys())):
            msg = f"{self.__class__.__name__} expected batch to contain keys {batch_fields} but found {batch.keys()}."
            raise ValueError(msg)
        field_dict = {field: batch.get(field) for field in batch if field in batch_fields}

        result = self.target(**field_dict)
        return result
