import torch

from replay.data.nn import TensorMap


class LengthBucketedQueryEncoder(torch.nn.Module):
    """Run similarly sized left-padded sequences together during training.

    The wrapper sorts rows by the number of ``True`` values in ``padding_mask``,
    splits them into buckets of at most ``bucket_size`` rows, and removes the
    left prefix not needed by the longest sequence in each bucket. It invokes
    ``encoder`` once per bucket and restores the original row order and sequence
    length afterwards. Removed positions in the returned tensor are zeros and
    must be excluded from the objective by the target mask.

    ``sequence_shift`` retains positions immediately before the first valid
    input. This is needed when targets are shifted relative to inputs: for a
    :class:`~replay.nn.transform.NextTokenTransform` with ``shift=1``, use
    ``sequence_shift=1``. It must match the target shift or positions required
    by the loss can be cropped.

    Bucketing is active only in training mode. In evaluation mode the full batch
    is passed to the wrapped encoder unchanged. The wrapped encoder must process
    rows independently and preserve the
    ``[batch, sequence, embedding]`` shape. Because rows are reordered and the
    encoder is called several times, stochastic layers can draw random values in
    a different order than without bucketing.

    Batch-aligned feature tensors are reordered; those whose second dimension is
    the sequence length are also cropped. Scalars and tensors not starting with
    the batch size are passed through unchanged. The attention mask can be
    shared (``[sequence, sequence]``), batch-aligned, or flattened as
    ``[batch * heads, sequence, sequence]``.

    For best efficiency, first use
    :class:`~replay.nn.transform.TargetAwareAdaptiveTrimTransform` to remove
    padding common to the physical batch, then use this wrapper to remove
    padding within each bucket.

    Example:

    .. code-block:: python

        from replay.nn.sequential import LengthBucketedQueryEncoder, SasRecTransformerLayer

        query_encoder = LengthBucketedQueryEncoder(
            encoder=SasRecTransformerLayer(
                embedding_dim=128,
                num_heads=4,
                num_blocks=2,
            ),
            bucket_size=32,
            sequence_shift=1,
        )
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        bucket_size: int,
        sequence_shift: int = 1,
    ) -> None:
        """
        :param encoder: Row-independent query encoder called as
            ``encoder(feature_tensors, input_embeddings, padding_mask, attention_mask)``.
        :param bucket_size: Maximum number of rows evaluated in one encoder call.
            A smaller value can remove more padding but causes more encoder calls.
        :param sequence_shift: Number of positions to retain immediately before
            the first valid input token. Must match the ``shift`` used by
            :class:`~replay.nn.transform.NextTokenTransform`.
        """
        super().__init__()
        if bucket_size <= 0:
            msg = "bucket_size must be positive"
            raise ValueError(msg)
        if sequence_shift < 0:
            msg = "sequence_shift must be non-negative"
            raise ValueError(msg)
        self.encoder = encoder
        self.bucket_size = bucket_size
        self.sequence_shift = sequence_shift

    def reset_parameters(self) -> None:
        reset_parameters = getattr(self.encoder, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    @staticmethod
    def _slice_features(
        feature_tensors: TensorMap,
        row_indices: torch.LongTensor,
        left_crop: int,
        batch_size: int,
        sequence_length: int,
    ) -> TensorMap:
        result = {}
        for name, value in feature_tensors.items():
            if value.ndim == 0 or value.size(0) != batch_size:
                result[name] = value
                continue
            if value.ndim >= 2 and value.size(1) == sequence_length:
                value = value[:, left_crop:]
            result[name] = value.index_select(0, row_indices)
        return result

    @staticmethod
    def _slice_attention_mask(
        attention_mask: torch.Tensor,
        row_indices: torch.LongTensor,
        left_crop: int,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        mask = attention_mask
        if mask.ndim < 2 or mask.shape[-2:] != (sequence_length, sequence_length):
            msg = "attention_mask must end with [sequence, sequence] dimensions."
            raise ValueError(msg)
        mask = mask[..., left_crop:, left_crop:]
        if mask.ndim == 2:
            return mask
        if mask.size(0) == batch_size:
            return mask.index_select(0, row_indices)
        if mask.ndim == 3 and mask.size(0) % batch_size == 0:
            num_heads = mask.size(0) // batch_size
            mask = mask.reshape(batch_size, num_heads, *mask.shape[-2:])
            return mask.index_select(0, row_indices).reshape(-1, *mask.shape[-2:])
        msg = "attention_mask must be shared, batch-aligned, or flattened as [batch * heads, sequence, sequence]."
        raise ValueError(msg)

    def forward(
        self,
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the wrapped encoder and restore the original batch layout.

        :param feature_tensors: Feature mapping consumed by the wrapped encoder.
        :param input_embeddings: Input tensor of shape
            ``[batch, sequence, embedding]``.
        :param padding_mask: Boolean mask of shape ``[batch, sequence]`` where
            ``True`` marks a valid input position.
        :param attention_mask: Attention mask ending in
            ``[sequence, sequence]`` dimensions.
        :return: Encoder output with the same shape as ``input_embeddings``.
        """
        if input_embeddings.ndim != 3:
            msg = "input_embeddings must have shape [batch, sequence, embedding]"
            raise ValueError(msg)
        if padding_mask.ndim != 2 or input_embeddings.shape[:2] != padding_mask.shape:
            msg = "padding_mask must match the first two input_embeddings dimensions"
            raise ValueError(msg)
        batch_size, sequence_length = padding_mask.shape
        if batch_size == 0:
            msg = "batch dimension must be non-empty"
            raise ValueError(msg)
        if not self.training:
            return self.encoder(feature_tensors, input_embeddings, padding_mask, attention_mask)

        # Synchronize lengths once instead of reading one device scalar per bucket.
        lengths = padding_mask.sum(dim=1, dtype=torch.int32)
        order = torch.argsort(lengths)
        sorted_lengths = lengths.index_select(0, order).cpu().tolist()
        output = input_embeddings.new_zeros(input_embeddings.shape)
        for start in range(0, batch_size, self.bucket_size):
            end = min(start + self.bucket_size, batch_size)
            row_indices = order[start:end]
            max_valid_length = sorted_lengths[end - 1]
            left_crop = 0 if max_valid_length == 0 else max(0, sequence_length - max_valid_length - self.sequence_shift)
            bucket_embeddings = input_embeddings[:, left_crop:].index_select(0, row_indices)
            bucket_padding_mask = padding_mask[:, left_crop:].index_select(0, row_indices)
            bucket_output = self.encoder(
                self._slice_features(
                    feature_tensors,
                    row_indices,
                    left_crop,
                    batch_size,
                    sequence_length,
                ),
                bucket_embeddings,
                bucket_padding_mask,
                self._slice_attention_mask(
                    attention_mask,
                    row_indices,
                    left_crop,
                    batch_size,
                    sequence_length,
                ),
            )
            if bucket_output.shape != bucket_embeddings.shape:
                msg = "The wrapped encoder must preserve [batch, sequence, embedding] shape."
                raise ValueError(msg)
            # Write buckets directly in source order without retaining padded copies.
            output[row_indices, left_crop:] = bucket_output

        return output
