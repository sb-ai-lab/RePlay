import torch


class TokenMaskTransform(torch.nn.Module):
    """
    For the feature tensor specified by ``token_name``, randomly masks items
    in the sequence based on a uniform distribution with specified probability of masking.
    In fact, this transform creates a mask for the Masked Language Modeling (MLM) task analog in the recommendations.

    Example:

    .. code-block:: python

        >>> _ = torch.manual_seed(0)
        >>> input_tensor = {"padding_id": torch.BoolTensor([0, 1, 1])}
        >>> transform = TokenMaskTransform("padding_id")
        >>> output_tensor = transform(input_tensor)
        >>> output_tensor
        {'padding_id': tensor([False,  True,  True]),
        'token_mask': tensor([False,  True, False])}

    """

    def __init__(
        self,
        token_name: str,
        out_feature_name: str = "token_mask",
        mask_prob: float = 0.15,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        :param token_name: A name of the column containing the unmasked tokes.
        :param out_feature_name: A name of the resulting  mask column. Default: ``token_mask``.
        :param mask_prob: Probability of masking the item, i.e. setting it to ``0``. Default: ``0.15``.
        :param generator: a random number generator to be used for generating
                the uniform distribution. Default: ``None``.
        """
        super().__init__()
        self.token_name = token_name
        self.out_feature_name = out_feature_name
        self.mask_prob = mask_prob
        self.generator = generator

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        paddings = batch[self.token_name]

        assert paddings.dtype == torch.bool, "Source tensor for token mask should be boolean."

        mask_prob = torch.rand(paddings.shape, dtype=torch.float32, device=paddings.device, generator=self.generator)

        # mask[i], 0 ~ mask_prob, 1 ~ (1 - mask_prob)
        mask = (mask_prob * paddings) >= self.mask_prob

        # Fix corner cases in mask
        # 1. If all token are not masked, add mask to the end
        if mask.all() or mask[paddings].all():
            mask[-1] = 0
        # 2. If all token are masked, add non-masked before the last
        elif (not mask.any()) and (len(mask) > 1):
            mask[-2] = 1

        output_batch[self.out_feature_name] = mask
        return output_batch


class TargetMLMTransform(torch.nn.Module):
    def __init__(
        self,
        token_mask_col: str = "token_mask",
        padding_mask_col: str = "padding_mask",
        out_feature_name: str = "target_padding_mask",
    ) -> None:
        super().__init__()
        self.token_mask_col = token_mask_col
        self.padding_mask_col = padding_mask_col
        self.out_feature_name = out_feature_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        output_batch[self.out_feature_name] = torch.logical_xor(
            output_batch[self.padding_mask_col], output_batch[self.token_mask_col]
        )
        return output_batch
