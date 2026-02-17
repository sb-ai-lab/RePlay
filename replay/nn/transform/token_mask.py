import torch


class TokenMaskTransform(torch.nn.Module):
    """
    For the feature tensor specified by ``token_field``, randomly masks items
    in the sequence based on a uniform distribution with specified probability of masking.
    In fact, this transform creates mask for the Masked Language Modeling (MLM) task analog in the recommendations.

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
        token_field: str,
        out_feature_name: str = "token_mask",
        mask_prob: float = 0.15,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        :param token_field: Name of the column containing the unmasked tokes.
        :param out_feature_name: Name of the resulting  mask column. Default: ``token_mask``.
        :param mask_prob: Probability of masking the item, i.e. setting it to ``0``. Default: ``0.15``.
        :param generator: Random number generator to be used for generating
                the uniform distribution. Default: ``None``.
        """
        super().__init__()
        self.token_field = token_field
        self.out_feature_name = out_feature_name
        self.mask_prob = mask_prob
        self.generator = generator

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        paddings = batch[self.token_field]

        assert paddings.dtype == torch.bool, "Source tensor for token mask should be boolean."

        mask_prob = torch.rand(paddings.size(-1), dtype=torch.float32, generator=self.generator).to(
            device=paddings.device
        )

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
