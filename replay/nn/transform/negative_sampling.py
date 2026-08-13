import torch


class UniformNegativeSamplingTransform(torch.nn.Module):
    """
    Transform for global negative sampling.

    For every batch, transform generates a vector of size ``(num_negative_samples)``
    consisting of random indices sampled from a range of ``cardinality``. Unless a custom sample
    distribution is provided, the indices are weighted equally.

    A one-dimensional distribution containing zero weights stores and samples only positive-weight
    items, then maps the result back to the original item IDs. Dense and multidimensional
    distributions retain their original representation.

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
        candidate_ids = None
        if sample_distribution.dim() == 1:
            if not sample_distribution.is_floating_point():
                msg = "sample_distribution must have a floating-point dtype."
                raise TypeError(msg)
            if not torch.isfinite(sample_distribution).all() or (sample_distribution < 0).any():
                msg = "sample_distribution must contain finite non-negative weights."
                raise ValueError(msg)
            positive_weight_mask = sample_distribution > 0
            positive_weight_count = torch.count_nonzero(positive_weight_mask).item()
            if positive_weight_count < num_negative_samples:
                msg = (
                    "sample_distribution must contain at least "
                    f"{num_negative_samples} positive-weight candidates, got {positive_weight_count}"
                )
                raise ValueError(msg)
            if positive_weight_count < cardinality:
                candidate_ids = torch.nonzero(positive_weight_mask, as_tuple=True)[0]
                sample_distribution = sample_distribution[candidate_ids].detach()

        self.register_buffer("_candidate_ids", candidate_ids)
        self.register_buffer("sample_distribution", sample_distribution)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        negatives = torch.multinomial(
            self.sample_distribution,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )
        if self._candidate_ids is not None:
            negatives = self._candidate_ids[negatives]

        output_batch[self.out_feature_name] = negatives
        return output_batch


class GroupedUniformNegativeSamplingTransform(UniformNegativeSamplingTransform):
    """Draw an independent negative pool for each logical batch group.

    This transform allows several logical batches to be packed into one larger
    physical batch without changing how often negatives are sampled.
    """

    def __init__(
        self,
        cardinality: int,
        num_negative_samples: int,
        group_size: int,
        groups_per_batch: int,
        *,
        batch_feature_name: str = "positive_labels",
        out_feature_name: str = "negative_labels",
        sample_distribution: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        :param cardinality: Number of items in the catalog, excluding padding.
        :param num_negative_samples: Number of items in each negative pool.
        :param group_size: Number of rows in one logical batch group.
        :param groups_per_batch: Number of logical groups in a full physical batch.
        :param batch_feature_name: Feature whose first dimension defines the physical batch size.
        :param out_feature_name: Name of the generated grouped negative tensor.
        :param sample_distribution: Optional sampling weights of length ``cardinality``.
        :param generator: Optional random number generator.
        """
        if group_size <= 0:
            msg = "The group_size parameter must be positive."
            raise ValueError(msg)
        if groups_per_batch <= 1:
            msg = "The groups_per_batch parameter must be greater than one."
            raise ValueError(msg)
        if sample_distribution is not None and sample_distribution.dim() != 1:
            msg = "Grouped sampling requires a one-dimensional sample_distribution."
            raise ValueError(msg)
        super().__init__(
            cardinality=cardinality,
            num_negative_samples=num_negative_samples,
            out_feature_name=out_feature_name,
            sample_distribution=sample_distribution,
            generator=generator,
        )
        self.group_size = group_size
        self.groups_per_batch = groups_per_batch
        self.batch_feature_name = batch_feature_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.batch_feature_name not in batch:
            msg = f"The batch does not contain {self.batch_feature_name!r}."
            raise ValueError(msg)
        batch_feature = batch[self.batch_feature_name]
        if batch_feature.dim() == 0 or batch_feature.size(0) == 0:
            msg = f"The {self.batch_feature_name!r} feature must have a non-empty batch dimension."
            raise ValueError(msg)

        active_groups = (batch_feature.size(0) + self.group_size - 1) // self.group_size
        if active_groups > self.groups_per_batch:
            capacity = self.group_size * self.groups_per_batch
            msg = f"Batch size {batch_feature.size(0)} exceeds grouped sampler capacity {capacity}."
            raise ValueError(msg)

        pools = torch.stack(
            [
                torch.multinomial(
                    self.sample_distribution,
                    num_samples=self.num_negative_samples,
                    replacement=False,
                    generator=self.generator,
                )
                for _ in range(active_groups)
            ]
        )
        if self._candidate_ids is not None:
            pools = self._candidate_ids[pools]
        if active_groups < self.groups_per_batch:
            # Keep a fixed output shape without advancing the RNG for groups the loss will ignore.
            pools = torch.cat(
                (pools, pools[-1:].expand(self.groups_per_batch - active_groups, -1)),
                dim=0,
            )

        output_batch = dict(batch.items())
        output_batch[self.out_feature_name] = pools
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
