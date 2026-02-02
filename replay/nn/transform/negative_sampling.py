import warnings
from typing import Literal, Optional, cast

import torch
import torch.nn.functional as func


class UniformNegativeSamplingTransform(torch.nn.Module):
    """
    Transform for global negative sampling.

    For every batch, transform generates a vector of size ``(num_negative_samples)``
    consisting of random indices sampeled from a range of ``cardinality``. Unless a custom sample
    distribution is provided, the indices are weighted equally.

    Example:

    .. code-block:: python

        >>> _ = torch.manual_seed(0)
        >>> input_batch = {"item_id": torch.LongTensor([[1, 0, 4]])}
        >>> transform = UniformNegativeSamplingTransform(cardinality=4, num_negative_samples=2)
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'item_id': tensor([[1, 0, 4]]), 'negative_labels': tensor([2, 1])}

    """

    def __init__(
        self,
        cardinality: int,
        num_negative_samples: int,
        *,
        out_feature_name: str = "negative_labels",
        sample_distribution: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        :param cardinality: number of unique items in vocabulary (catalog).
            The specified cardinality value must not take into account the padding value.
        :param num_negative_samples: The size of negatives vector to generate.
        :param out_feature_name: The name of result feature in batch.
        :param sample_distribution: The weighs of indices in the vocabulary. If specified, must
                match the ``cardinality``. Default: ``None``.
        :param generator: Random number generator to be used for sampling
                from the distribution. Default: ``None``.
        """
        if sample_distribution is not None:
            if sample_distribution.ndim != 1:
                msg: str = (
                    f"The `sample_distribution` parameter must be 1D.Got {sample_distribution.ndim}, will be flattened."
                )
                warnings.warn(msg)
                sample_distribution = sample_distribution.flatten()
            if sample_distribution.size(-1) != cardinality:
                msg: str = (
                    "The sample_distribution parameter has an incorrect size. "
                    f"Got {sample_distribution.size(-1)}, expected {cardinality}."
                )
                raise ValueError(msg)

        if num_negative_samples >= cardinality:
            msg = (
                "The `num_negative_samples` parameter has an incorrect value."
                f"Got {num_negative_samples}, expected less than cardinality of items catalog ({cardinality})."
            )
            raise ValueError(msg)

        super().__init__()

        self.out_feature_name = out_feature_name
        self.num_negative_samples = num_negative_samples
        self.generator = generator
        sample_distribution = sample_distribution if sample_distribution is not None else torch.ones(cardinality)
        self.sample_distribution = torch.nn.Buffer(cast(torch.Tensor, sample_distribution))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        negatives = torch.multinomial(
            self.sample_distribution,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        device = next(iter(output_batch.values())).device
        output_batch[self.out_feature_name] = negatives.to(device)
        return output_batch


class FrequencyNegativeSamplingTransform(torch.nn.Module):
    """
    Transform for global negative sampling.

    For every batch, transform generates a vector of size ``(num_negative_samples)``
    consisting of random indices sampeled from a range of ``cardinality``.

    Indices frequency will be computed and their sampling will be done
    according to their respective frequencies.

    Example:

    .. code-block:: python

        >>> _ = torch.manual_seed(0)
        >>> input_batch = {"item_id": torch.LongTensor([[1, 0, 4]])}
        >>> transform = FrequencyNegativeSamplingTransform(cardinality=4, num_negative_samples=2)
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'item_id': tensor([[1, 0, 4]]), 'negative_labels': tensor([2, 1])}

    """

    def __init__(
        self,
        cardinality: int,
        num_negative_samples: int,
        *,
        out_feature_name: str = "negative_labels",
        generator: Optional[torch.Generator] = None,
        mode: Literal["softmax", "softsum"] = "softmax",
    ) -> None:
        """
        :param cardinality: The size of sample vocabulary.
        :param num_negative_samples: The size of negatives vector to generate.
        :param out_feature_name: The name of result feature in batch.
        :param generator: Random number generator to be used for sampling
                from the distribution. Default: ``None``.
        :param mode: Mode of frequency-based samping for undersampled items.
            Default: ``softmax``.
        """
        assert num_negative_samples < cardinality

        super().__init__()

        self.cardinality = cardinality
        self.out_feature_name = out_feature_name
        self.num_negative_samples = num_negative_samples
        self.generator = generator
        self.mode = mode

        self.frequencies = torch.nn.Buffer(torch.zeros(cardinality, dtype=torch.int64))

    def get_probas(self) -> torch.Tensor:
        raw: torch.Tensor = 1.0 / (1.0 + self.frequencies)
        if self.mode == "softsum":
            result: torch.Tensor = raw / torch.sum(raw)
        elif self.mode == "softmax":
            result: torch.Tensor = func.softmax(raw, dim=-1)
        else:
            msg: str = f"Unsupported mode: {self.mode}."
            raise TypeError(msg)
        return result

    def update_probas(self, selected: torch.Tensor) -> None:
        device = self.frequencies.device
        one = torch.ones(1, dtype=torch.int64, device=device)
        self.frequencies.index_add_(-1, selected, one.expand(selected.numel()))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        negatives = torch.multinomial(
            input=self.get_probas(),
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        self.update_probas(negatives)

        device = next(iter(output_batch.values())).device
        output_batch[self.out_feature_name] = negatives.to(device)
        return output_batch


class ThresholdNegativeSamplingTransform(torch.nn.Module):
    """
    Transform for global negative sampling.

    For every batch, transform generates a vector of size ``(num_negative_samples)``
    consisting of random indices sampeled from a range of ``cardinality``.

    Indices that are oversampled at this point will be ignored, while
    other samples will be chosen according to their respective frequency.

    Example:

    .. code-block:: python

        >>> _ = torch.manual_seed(0)
        >>> input_batch = {"item_id": torch.LongTensor([[1, 0, 4]])}
        >>> transform = ThresholdNegativeSamplingTransform(cardinality=4, num_negative_samples=2)
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'item_id': tensor([[1, 0, 4]]), 'negative_labels': tensor([2, 1])}

    """

    def __init__(
        self,
        cardinality: int,
        num_negative_samples: int,
        *,
        out_feature_name: str = "negative_labels",
        generator: Optional[torch.Generator] = None,
        mode: Literal["softmax", "softsum"] = "softmax",
    ) -> None:
        """
        :param cardinality: The size of sample vocabulary.
        :param num_negative_samples: The size of negatives vector to generate.
        :param out_feature_name: The name of result feature in batch.
        :param generator: Random number generator to be used for sampling
                from the distribution. Default: ``None``.
        :param mode: Mode of frequency-based samping for undersampled items.
            Default: ``softmax``.
        """
        assert num_negative_samples < cardinality

        super().__init__()

        self.cardinality = cardinality
        self.out_feature_name = out_feature_name
        self.num_negative_samples = num_negative_samples
        self.generator = generator
        self.mode = mode

        self.frequencies = torch.nn.Buffer(torch.zeros(cardinality, dtype=torch.int64))

    def get_probas(self) -> torch.Tensor:
        raw: torch.Tensor = 1.0 / (1.0 + self.frequencies)
        thr: torch.Tensor = torch.max(self.frequencies)
        mask: torch.Tensor = thr != self.frequencies
        if self.mode == "softsum":
            eps = torch.finfo(raw.dtype).eps
            raw = torch.where(mask, raw, eps)
            result: torch.Tensor = raw / torch.sum(raw)
        elif self.mode == "softmax":
            inf = torch.finfo(raw.dtype).min
            raw = torch.where(mask, raw, inf)
            result: torch.Tensor = func.softmax(raw, dim=-1)
        else:
            msg: str = f"Unsupported mode: {self.mode}."
            raise TypeError(msg)
        return result

    def update_probas(self, selected: torch.Tensor) -> None:
        device = self.frequencies.device
        one = torch.ones(1, dtype=torch.int64, device=device)
        self.frequencies.index_add_(-1, selected, one.expand(selected.numel()))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        negatives = torch.multinomial(
            input=self.get_probas(),
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        self.update_probas(negatives)

        device = next(iter(output_batch.values())).device
        output_batch[self.out_feature_name] = negatives.to(device)
        return output_batch


class MultiClassNegativeSamplingTransform(torch.nn.Module):
    """
    Transform for generating negatives using a fixed class-assignment matrix.

    For every batch, transform generates a tensor of size ``(N, num_negative_samples)``, where N is number of classes.
    This tensor consists of random indices sampled using specified fixed class-assignment matrix.

    Also, transform receives from batch by key a tensor ``negative_selector_name`` of shape (batch size,),
    where i-th element in [0, N-1] specifies which class of N is used to select from sampled negatives that corresponds
    to every i-th batch row (user's history sequence).

    The resulting negatives tensor has shape of ``(batch_size, num_negative_samples)``.

    Example:

    .. code-block:: python

        >>> _ = torch.manual_seed(0)
        >>> sample_mask = torch.tensor([
        ...     [1, 0, 1, 0, 0, 0],
        ...     [0, 0, 0, 1, 1, 0],
        ...     [0, 1, 0, 0, 0, 1],
        ... ])
        >>> input_batch = {"negative_selector": torch.tensor([0, 2, 1, 1, 0])}
        >>> transform = MultiClassNegativeSamplingTransform(
        ...                 num_negative_samples=2,
        ...                 sample_mask=sample_mask
        ... )
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'negative_selector': tensor([0, 2, 1, 1, 0]),
         'negative_labels': tensor([[2, 0],
                 [5, 1],
                 [3, 4],
                 [3, 4],
                 [2, 0]])}
    """

    def __init__(
        self,
        num_negative_samples: int,
        sample_mask: torch.Tensor,
        *,
        negative_selector_name: str = "negative_selector",
        out_feature_name: str = "negative_labels",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        :param num_negative_samples: The size of negatives vector to generate.
        :param sample_mask: The class-assignment (indicator) matrix of shape: ``(N, number of items in catalog)``,
            where ``sample_mask[n, i]`` is a weight (or binary indicator) of assigning item i to class n.
        :param negative_selector_name: name of tensor in batch of shape (batch size,), where i-th element
            in [0, N-1] specifies which class of N is used to get negatives corresponding to i-th ``query_id`` in batch.
        :param out_feature_name: The name of result feature in batch.
        :param generator: Random number generator to be used for sampling from the distribution. Default: ``None``.
        """
        if sample_mask.dim() != 2:
            msg = (
                "The `sample_mask` parameter has an incorrect shape."
                f"Got {sample_mask.dim()}, expected shape: (number of classes, number of items in catalog)."
            )
            raise ValueError(msg)

        if num_negative_samples >= sample_mask.size(-1):
            msg = (
                "The `num_negative_samples` parameter has an incorrect value."
                f"Got {num_negative_samples}, expected less than cardinality of items catalog ({sample_mask.size(-1)})."
            )
            raise ValueError(msg)

        super().__init__()

        self.sample_mask = torch.nn.Buffer(sample_mask.float())

        self.num_negative_samples = num_negative_samples
        self.negative_selector_name = negative_selector_name
        self.out_feature_name = out_feature_name
        self.generator = generator

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.negative_selector_name in batch
        assert batch[self.negative_selector_name].dim() == 1

        negative_selector = batch[self.negative_selector_name]  # [batch_size]

        # [N, num_negatives] - shape of negatives
        negatives = torch.multinomial(
            input=self.sample_mask,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        # [N, num_negatives] -> [batch_size, num_negatives]
        selected_negatives = negatives[negative_selector]

        output_batch = dict(batch.items())
        output_batch[self.out_feature_name] = selected_negatives.to(device=negative_selector.device)
        return output_batch
