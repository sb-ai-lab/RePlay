import torch


def _validated_sample_distribution(
    cardinality: int,
    sample_distribution: torch.Tensor | None,
) -> torch.Tensor:
    if sample_distribution is None:
        return torch.ones(cardinality)
    if sample_distribution.ndim != 1:
        msg = "sample_distribution must be a one-dimensional tensor."
        raise ValueError(msg)
    if sample_distribution.numel() != cardinality:
        msg = (
            "The sample_distribution parameter has an incorrect size. "
            f"Got {sample_distribution.numel()}, expected {cardinality}."
        )
        raise ValueError(msg)
    if not sample_distribution.is_floating_point():
        msg = "sample_distribution must have a floating-point dtype."
        raise TypeError(msg)
    if not torch.isfinite(sample_distribution).all() or (sample_distribution < 0).any():
        msg = "sample_distribution must contain finite non-negative weights."
        raise ValueError(msg)
    return sample_distribution


class UniformNegativeSamplingTransform(torch.nn.Module):
    """
    Transform for global negative sampling.

    For every batch, transform generates a vector of size ``(num_negative_samples)``
    consisting of random indices sampled from a range of ``cardinality``. Unless a custom sample
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
        out_feature_name: str | None = "negative_labels",
        sample_distribution: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        :param cardinality: number of unique items in the vocabulary (or catalog).
            The specified cardinality value must not take into account the padding value.
        :param num_negative_samples: The size of negatives vector to generate.
        :param out_feature_name: The name of the result feature in a batch.
        :param sample_distribution: The weighs of indices in the vocabulary. If specified, must
                match the ``cardinality``. Default: ``None``.
        :param generator: a random number generator to be used for sampling
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

        if sample_distribution is None:
            sample_distribution = torch.ones(cardinality)
        self.register_buffer("sample_distribution", sample_distribution)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        negatives = torch.multinomial(
            self.sample_distribution,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        output_batch[self.out_feature_name] = negatives
        return output_batch


class SparseUniformNegativeSamplingTransform(torch.nn.Module):
    """
    Sample unique negatives from items with positive sampling weights.

    Unlike :class:`UniformNegativeSamplingTransform`, this transform stores and samples only
    the positive-weight part of a sparse distribution. Sampled positions are mapped back to
    catalog item IDs before they are added to the batch. A dense distribution uses the original
    representation without an additional item mapping.
    """

    def __init__(
        self,
        cardinality: int,
        num_negative_samples: int,
        *,
        out_feature_name: str | None = "negative_labels",
        sample_distribution: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        :param cardinality: number of unique items in the catalog, excluding padding.
        :param num_negative_samples: number of unique negative item IDs to generate.
        :param out_feature_name: name of the generated batch feature.
        :param sample_distribution: one-dimensional, non-negative sampling weights for the
            complete catalog. Zero-weight items are removed from the stored distribution.
        :param generator: random number generator used for sampling.
        """
        if cardinality <= 0:
            msg = "cardinality must be positive."
            raise ValueError(msg)
        if num_negative_samples <= 0:
            msg = "num_negative_samples must be positive."
            raise ValueError(msg)
        if num_negative_samples >= cardinality:
            msg = (
                "The num_negative_samples parameter has an incorrect value. "
                f"Got {num_negative_samples}, expected less than catalog cardinality ({cardinality})."
            )
            raise ValueError(msg)
        sample_distribution = _validated_sample_distribution(cardinality, sample_distribution)

        positive_weight_count = torch.count_nonzero(sample_distribution).item()
        if positive_weight_count < num_negative_samples:
            msg = (
                "sample_distribution must contain at least "
                f"{num_negative_samples} positive-weight candidates, got {positive_weight_count}"
            )
            raise ValueError(msg)

        super().__init__()
        self.out_feature_name = out_feature_name
        self.num_negative_samples = num_negative_samples
        self.generator = generator
        if positive_weight_count == cardinality:
            candidate_ids = None
            compact_distribution = sample_distribution
        else:
            candidate_ids = torch.nonzero(sample_distribution > 0, as_tuple=True)[0]
            compact_distribution = sample_distribution[candidate_ids].contiguous()
        self.register_buffer("candidate_ids", candidate_ids)
        self.register_buffer("sample_distribution", compact_distribution.detach())

    def _validate_generator_device(self) -> None:
        if self.generator is None:
            return
        generator_device = torch.device(self.generator.device)
        distribution_device = self.sample_distribution.device
        devices_are_compatible = generator_device.type == distribution_device.type and (
            generator_device.type != "cuda"
            or generator_device.index is None
            or distribution_device.index is None
            or generator_device.index == distribution_device.index
        )
        if not devices_are_compatible:
            msg = (
                "generator and sample_distribution must be on the same device; "
                f"got {generator_device} and {distribution_device}. Replace generator after moving the transform."
            )
            raise RuntimeError(msg)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add a shared unique negative pool to a shallow batch copy."""
        self._validate_generator_device()
        compact_ids = torch.multinomial(
            self.sample_distribution,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        output_batch = dict(batch)
        output_batch[self.out_feature_name] = (
            compact_ids if self.candidate_ids is None else torch.take(self.candidate_ids, compact_ids)
        )
        return output_batch


class MultiClassNegativeSamplingTransform(torch.nn.Module):
    """
    Transform for generating negatives using a fixed class-assignment matrix.

    For every batch, transform generates a tensor of size ``(N, num_negative_samples)``,
    where N is the number of classes.
    This tensor consists of random indices sampled using the specified fixed class-assignment matrix.

    Also, transform receives a tensor `negative_selector_name` of shape (batch size,) from a batch by the key,
    where the i-th element in [0, N-1] specifies which class of N is used to select from sampled negatives
    that correspond to every i-th batch row (user's history sequence).

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
        negative_selector_name: str | None = "negative_selector",
        out_feature_name: str | None = "negative_labels",
        generator: torch.Generator | None = None,
    ) -> None:
        """
        :param num_negative_samples: The size of negatives vector to generate.
        :param sample_mask: The class-assignment (indicator) matrix of shape: ``(N, number of items in catalog)``,
            where ``sample_mask[n, i]`` is a weight (or binary indicator) of assigning item i to class n.
        :param negative_selector_name: a name of a tensor in a batch of shape (batch size,), where the i-th element
            in [0, N-1] specifies which class of N is used to get negatives corresponding to i-th ``query_id`` in batch.
        :param out_feature_name: The name of the result feature in a batch.
        :param generator: a random number generator to be used for sampling from the distribution. Default: ``None``.
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

        output_batch = dict(batch.items())

        negative_selector = output_batch[self.negative_selector_name]  # [batch_size]

        # [N, num_negatives] - shape of negatives
        negatives = torch.multinomial(
            input=self.sample_mask,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        # [N, num_negatives] -> [batch_size, num_negatives]
        output_batch[self.out_feature_name] = negatives[negative_selector]
        return output_batch
