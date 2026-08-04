import torch

from replay.data.nn import TensorMap


class LengthBucketedQueryEncoder(torch.nn.Module):
    """Evaluate similarly sized left-padded sequences together during training.

    Cropped padding positions are restored as zeros. Downstream training code
    must ignore them through its target mask. RePlay attention masks may be
    shared, batch-aligned, or flattened as ``[batch * heads, sequence, sequence]``.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        bucket_size: int,
        prediction_shift: int = 1,
        bucket_during_eval: bool = False,
        min_input_elements: int = 4_194_304,
    ) -> None:
        """
        :param encoder: query encoder accepting feature tensors, embeddings, padding mask and attention mask.
        :param bucket_size: maximum number of rows evaluated together.
        :param prediction_shift: number of positions retained before the first valid input token.
        :param bucket_during_eval: whether to use bucketing outside training.
        :param min_input_elements: minimum number of embedding elements required to use bucketing.
            Set to ``0`` to enable bucketing for every batch.
        """
        super().__init__()
        if bucket_size <= 0:
            msg = "bucket_size must be positive"
            raise ValueError(msg)
        if prediction_shift < 0:
            msg = "prediction_shift must be non-negative"
            raise ValueError(msg)
        if min_input_elements < 0:
            msg = "min_input_elements must be non-negative"
            raise ValueError(msg)
        self.encoder = encoder
        self.bucket_size = bucket_size
        self.prediction_shift = prediction_shift
        self.bucket_during_eval = bucket_during_eval
        self.min_input_elements = min_input_elements

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
            value = value.index_select(0, row_indices)
            if value.ndim >= 2 and value.size(1) == sequence_length:
                value = value[:, left_crop:]
            result[name] = value
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
        """Run the wrapped encoder and restore the original row order."""
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
        if (not self.training and not self.bucket_during_eval) or (input_embeddings.numel() < self.min_input_elements):
            return self.encoder(feature_tensors, input_embeddings, padding_mask, attention_mask)

        lengths = padding_mask.sum(dim=1, dtype=torch.int32).detach().cpu().tolist()
        order_list = sorted(range(batch_size), key=lengths.__getitem__)
        order = torch.tensor(order_list, dtype=torch.long, device=padding_mask.device)
        outputs = []
        for start in range(0, batch_size, self.bucket_size):
            end = min(start + self.bucket_size, batch_size)
            row_indices = order[start:end]
            max_valid_length = lengths[order_list[end - 1]]
            left_crop = (
                0 if max_valid_length == 0 else max(0, sequence_length - max_valid_length - self.prediction_shift)
            )
            bucket_embeddings = input_embeddings.index_select(0, row_indices)[:, left_crop:]
            bucket_output = self.encoder(
                self._slice_features(
                    feature_tensors,
                    row_indices,
                    left_crop,
                    batch_size,
                    sequence_length,
                ),
                bucket_embeddings,
                padding_mask.index_select(0, row_indices)[:, left_crop:],
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
            if left_crop:
                prefix = bucket_output.new_zeros(bucket_output.size(0), left_crop, bucket_output.size(2))
                bucket_output = torch.cat((prefix, bucket_output), dim=1)
            outputs.append(bucket_output)

        sorted_output = torch.cat(outputs, dim=0)
        return sorted_output.index_select(0, torch.argsort(order))
