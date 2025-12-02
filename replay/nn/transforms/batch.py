from typing import Generic, NamedTuple, TypeVar, Union

import torch
from typing_extensions import Self

from replay.nn.transforms.base import BaseTransform

T = TypeVar("T", bound=NamedTuple)


class BatchingTransform(BaseTransform, Generic[T]):
    """
    Transforms a batch dict into an specified NamedTuple of features.
    For example, `SasRecTrainingBatch` may be a target NamedTuple for converting source batch.

    Example:

    .. code-block:: python

        >>> input_batch = {
        >>>         "query_id": torch.LongTensor([[0]]),
        >>>         "padding_mask": torch.BoolTensor([[0, 1, 1]]),
        >>>         "features": {"item_id": torch.LongTensor([[30, 22, 1]])}
        >>> }
        >>> transform = BatchingTransform(SasRecPredictionBatch)
        >>> output_batch = transform(input_batch)
        >>> print(output_batch)
        SasRecPredictionBatch(
            query_id=tensor([[0]]),
            padding_mask=tensor([[False,  True,  True]]),
            features={'item_id': tensor([[30, 22,  1]])})

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
