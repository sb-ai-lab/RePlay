import torch


class TokenMaskTransform(torch.nn.Module):
    """
    For the padding mask tensor specified by ``token_name``, randomly masks items
    in the sequence based on a uniform distribution with specified probability of masking.

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
        :param mask_prob: Probability of masking the item. Default: ``0.15``.
        :param generator: a random number generator to be used for generating
                the uniform distribution. Default: ``None``.
        """
        super().__init__()
        if not (0 < mask_prob < 1):
            msg = "Probability of masking must be 0 < `mask_prob` < 1"
            raise ValueError(msg)
        self.token_name = token_name
        self.out_feature_name = out_feature_name
        self.mask_prob = mask_prob
        self.generator = generator

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        paddings = batch[self.token_name]

        assert paddings.dtype == torch.bool, "Source tensor for token mask should be boolean."

        mask_prob = torch.rand(paddings.shape, dtype=torch.float32, generator=self.generator).to(paddings.device)

        # mask[i], 0 ~ mask_prob, 1 ~ (1 - mask_prob)
        mask = paddings * (mask_prob >= self.mask_prob)

        # Fix corner cases in mask
        # 1. If all token are not masked, add mask to the end
        valid_mask = mask[paddings]
        if valid_mask.all():
            mask[-1, -1] = False
        # 2. If all token are masked, add non-masked before the last
        elif not valid_mask.any():
            mask[-1, -2] = True

        output_batch[self.out_feature_name] = mask
        return output_batch


class TargetMLMTransform(torch.nn.Module):
    """
    Creates a target mask for the recsys analog of Masked Language Modeling (MLM) task.

    The resulting mask selects non-padding tokens masked by :class:`TokenMaskTransform`.
    """

    def __init__(
        self,
        token_mask_col: str = "token_mask",
        padding_mask_col: str = "padding_mask",
        out_feature_name: str = "target_padding_mask",
    ) -> None:
        """
        :param token_mask_col: A name of the column containing the token mask.
        :param padding_mask_col: A name of the column containing the padding mask.
        :param out_feature_name: A name of the resulting target mask column.
        """
        super().__init__()
        self.token_mask_col = token_mask_col
        self.padding_mask_col = padding_mask_col
        self.out_feature_name = out_feature_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        output_batch[self.out_feature_name] = output_batch[self.padding_mask_col] & ~output_batch[self.token_mask_col]
        return output_batch
