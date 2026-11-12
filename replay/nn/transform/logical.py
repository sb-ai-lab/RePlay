from collections.abc import Callable
from typing import Any

import torch


class FunctionTransform(torch.nn.Module):
    """Apply a function to a tensor from a batch.

    The transform can be used with unary functions, such as
    :func:`torch.logical_not`, and binary functions, such as
    :func:`torch.logical_and`. For a binary function, ``other`` can be the
    name of another tensor in the batch or a constant value.
    The input batch is not modified. If ``out`` is omitted, the input tensor
    is replaced in the returned batch.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "left": torch.tensor([True, True, False]),
        ...     "right": torch.tensor([True, False, False]),
        ... }
        >>> transform = FunctionTransform(torch.logical_and, "left", other="right", out="both")
        >>> transform(input_batch)
        {'left': tensor([ True,  True, False]), 'right': tensor([ True, False, False]),
         'both': tensor([ True, False, False])}

    """

    def __init__(
        self,
        function: Callable[..., torch.Tensor],
        input: str,
        other: str | None = None,
        out: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        :param function: Function applied to the selected tensors. The input
            tensor is passed as its first positional argument.
        :param input: Name of the input tensor in the batch.
        :param other: Optional name of second input tensor.
        :param out: Name under which to store the result. If ``None``, the
            result replaces ``input`` in the returned batch.
        :param kwargs: Additional keyword arguments passed to ``function``.
        """
        super().__init__()

        if not callable(function):
            msg = "The function parameter must be callable."
            raise TypeError(msg)

        self.function = function
        self.input = input
        self.other = other
        self.out = input if out is None else out
        self.kwargs = kwargs

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        function_args = [output_batch[self.input]]
        if self.other is not None:
            function_args.append(output_batch[self.other])

        output_batch[self.out] = self.function(*function_args, **self.kwargs)
        return output_batch
