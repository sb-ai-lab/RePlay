from typing import Optional

import torch

from replay.data.nn.transforms.base import BaseTransform


class TokenMaskTransform(BaseTransform):
    """
    For the feature tensor specified by ``token_field``, randomly masks items
    in the sequence based on a uniform distribution with specified probability of masking.
    In fact, this transform creates mask for the MLM task analog in the recommendations.
    Requiered for BERT4Rec model.
    """

    def __init__(
        self,
        token_field: str,
        mask_name: str = "token_mask",
        mask_prob: float = 0.15,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        :param token_field: Name of the column containing the unmasked tokes.
        :param mask_name: Name ofthe resulting  mask column. Default: ``token_mask``.
        :param mask_prob: Probability of masking the item, i.e. setting it to 0. Default: ``0.15``.
        :param generator: Random number generator to be used for generatring
                the uniform distribution. Default: ``None``.
        """
        super().__init__()
        self.token_field = token_field
        self.mask_name = mask_name
        self.mask_prob = mask_prob
        self.generator = generator

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

        batch[self.mask_name] = mask
        return batch
