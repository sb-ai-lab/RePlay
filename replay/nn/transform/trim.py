import torch


class TrimTransform(torch.nn.Module):
    """
    Trims sequences of specified names `feature_names` keeping the specified sequence length `seq_len` on the right.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "user_id": torch.LongTensor([111]),
        ...     "item_id": torch.LongTensor([[5, 4, 0, 7, 4]]),
        ...     "seen_ids": torch.LongTensor([[5, 4, 0, 7, 4]]),
        ... }
        >>> transform = TrimTransform(seq_len=3, feature_names="item_id")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'user_id': tensor([111]),
        'item_id': tensor([[0, 7, 4]]),
        'seen_ids': tensor([[5, 4, 0, 7, 4]])}

    """

    def __init__(
        self,
        seq_len: int,
        feature_names: list[str] | str,
    ) -> None:
        """
        :param seq_len: a maximum sequence length used in a model.
        :param feature_name: a name of a feature in a batch to be trimmed.
        """
        super().__init__()
        assert seq_len > 0
        self.seq_len = seq_len
        self.feature_names = [feature_names] if isinstance(feature_names, str) else feature_names

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        for name in self.feature_names:
            assert output_batch[name].shape[1] >= self.seq_len
            output_batch[name] = output_batch[name][:, -self.seq_len :, ...].contiguous()

        return output_batch


class AdaptiveTrimTransform(torch.nn.Module):
    """
    Trims sequences of the specified names `feature_names` to the maximum sequence length in the current batch.
    This transform is intended for use during validation and inference to speed up processing by reducing
    the length of padded sequence parts. Note that sequences should be left-padded.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "item_id": torch.LongTensor([[5, 5, 5, 5, 0], [5, 5, 0, 2, 4]]),
        ...     "padding_mask": torch.BoolTensor([[0, 0, 0, 0, 1], [0, 0, 1, 1, 1]]),
        ... }
        >>> transform = AdaptiveTrimTransform("item_id", padding_mask_name="padding_mask")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'item_id': tensor([[5, 5, 0],
                 [0, 2, 4]]),
         'padding_mask': tensor([[False, False,  True],
                 [ True,  True,  True]])}


    """

    def __init__(
        self,
        feature_names: list[str] | str,
        padding_mask_name: str = "padding_mask",
    ) -> None:
        """
        :param feature_name: name of features in batch to be trimmed.
            `padding_mask_name` will be included automatically.
        :param padding_mask_name: name of padding_mask in batch.
        """
        super().__init__()
        self.feature_names = [feature_names] if isinstance(feature_names, str) else feature_names.copy()
        if padding_mask_name not in self.feature_names:
            self.feature_names.append(padding_mask_name)

        self.padding_mask_name = padding_mask_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.padding_mask_name not in batch:
            msg = f"Padding mask '{self.padding_mask_name}' not found in batch."
            raise KeyError(msg)

        assert batch[self.padding_mask_name].ndim == 2
        source_seqlen = batch[self.padding_mask_name].size(1)
        max_non_padded_seqlen = batch[self.padding_mask_name].sum(dim=1).max().item()

        if max_non_padded_seqlen == source_seqlen:
            return batch

        output_batch = dict(batch.items())
        for name in self.feature_names:
            output_batch[name] = output_batch[name][:, -max_non_padded_seqlen:, ...].contiguous()
        return output_batch


class TargetAwareAdaptiveTrimTransform(torch.nn.Module):
    """Remove the common left prefix unused by inputs and training targets.

    The transform is intended for left-padded sequential batches. A column can
    be removed only when it is ``False`` in both ``padding_mask`` and
    ``target_padding_mask`` for every row. This target-aware rule preserves the
    extra position commonly introduced by next-token shifting, even when that
    position is padding from the model-input point of view.

    Every tensor named in ``feature_names``, as well as both masks, must start
    with the same ``[batch, sequence]`` dimensions. They are sliced together on
    the sequence dimension; trailing dimensions are preserved. Other batch
    entries are left unchanged. Both masks must be two-dimensional boolean
    tensors on the same device. If there is no removable prefix, the original
    batch object is returned.

    Apply this transform after creating the input and target masks, but before
    adding a positive-label dimension to ``target_padding_mask``. It should be
    combined with :class:`~replay.nn.sequential.LengthBucketedQueryEncoder`:
    this transform crops padding shared by the physical batch, and the encoder
    then crops additional padding inside groups of similarly sized sequences.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "item_id": torch.tensor([[0, 0, 3, 4, 5]]),
        ...     "positive_labels": torch.tensor([[0, 3, 4, 5, 6]]),
        ...     "padding_mask": torch.tensor([[False, False, True, True, True]]),
        ...     "target_padding_mask": torch.tensor([[False, True, True, True, True]]),
        ... }
        >>> transform = TargetAwareAdaptiveTrimTransform(
        ...     feature_names=["item_id", "positive_labels"],
        ... )
        >>> output_batch = transform(input_batch)
        >>> output_batch["item_id"]
        tensor([[0, 3, 4, 5]])
        >>> output_batch["target_padding_mask"]
        tensor([[True, True, True, True]])
    """

    def __init__(
        self,
        feature_names: list[str] | str,
        padding_mask_name: str = "padding_mask",
        target_padding_mask_name: str = "target_padding_mask",
    ) -> None:
        """
        :param feature_names: Name or names of sequence tensors to trim. The two
            mask names are added automatically and need not be included.
        :param padding_mask_name: Name of the boolean input mask, where ``True``
            marks a valid model-input position.
        :param target_padding_mask_name: Name of the boolean target mask, where
            ``True`` marks a valid training target.
        """
        super().__init__()
        names = [feature_names] if isinstance(feature_names, str) else list(feature_names)
        self.feature_names = list(dict.fromkeys((*names, padding_mask_name, target_padding_mask_name)))
        self.padding_mask_name = padding_mask_name
        self.target_padding_mask_name = target_padding_mask_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        padding_mask = batch[self.padding_mask_name]
        target_padding_mask = batch[self.target_padding_mask_name]
        if padding_mask.ndim != 2 or target_padding_mask.ndim != 2:
            msg = "Input and target padding masks must have shape [batch, sequence]."
            raise ValueError(msg)
        if padding_mask.shape != target_padding_mask.shape:
            msg = "Input and target padding masks must have equal shapes."
            raise ValueError(msg)
        if padding_mask.dtype != torch.bool or target_padding_mask.dtype != torch.bool:
            msg = "Input and target padding masks must have boolean dtype."
            raise TypeError(msg)
        if padding_mask.device != target_padding_mask.device:
            msg = "Input and target padding masks must be on the same device."
            raise ValueError(msg)
        for name in self.feature_names:
            if batch[name].shape[:2] != padding_mask.shape:
                msg = f"Feature '{name}' must start with shape {tuple(padding_mask.shape)}."
                raise ValueError(msg)

        active_columns = (padding_mask | target_padding_mask).any(dim=0)
        trim_start = int(active_columns.long().argmax().item())
        if trim_start == 0:
            return batch

        output_batch = dict(batch.items())
        for name in self.feature_names:
            output_batch[name] = output_batch[name][:, trim_start:, ...].contiguous()
        return output_batch
