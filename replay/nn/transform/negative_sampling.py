from typing import Optional

import torch


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
        out_feature_name: Optional[str] = "negative_labels",
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
        if sample_distribution is not None and sample_distribution.size(-1) != cardinality:
            msg = (
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
        if sample_distribution is not None:
            self.sample_distribution = sample_distribution
        else:
            self.sample_distribution = torch.ones(cardinality)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        negatives = torch.multinomial(
            self.sample_distribution,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        output_batch[self.out_feature_name] = negatives.to(device=next(iter(output_batch.values())).device)
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
        ...     [1, 0, 1, 0, 0, 1],
        ...     [0, 0, 0, 1, 1, 0],
        ... ])
        >>> input_batch = {"negative_selector": torch.tensor([0, 0, 1, 1, 0])}
        >>> transform = MultiClassNegativeSamplingTransform(
        ...                 num_negative_samples=2,
        ...                 sample_mask=sample_mask
        ... )
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'negative_selector': tensor([0, 0, 1, 1, 0]),
         'negative_labels': tensor([[2, 5],
                 [2, 5],
                 [3, 4],
                 [3, 4],
                 [2, 5]])}
    """

    def __init__(
        self,
        num_negative_samples: int,
        sample_mask: torch.Tensor,
        *,
        negative_selector_name: Optional[str] = "negative_selector",
        out_feature_name: Optional[str] = "negative_labels",
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

        self.register_buffer("sample_mask", sample_mask.float())

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
