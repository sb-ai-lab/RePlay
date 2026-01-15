from typing import Optional

import torch

from replay.nn.transforms.base import BaseTransform


class UniformNegativeSamplingTransform(BaseTransform):
    """
    Batch-independent negative sampling.

    For every batch, transform generates a vector of size ``(num_negative_samples)``
    consisting of random indices sampeled from a range of ``vocab_size``. Unless a custom sample
    distribution is provided, the indices are weighted equally.

    Example:

    .. code-block:: python

        >>> _ = torch.manual_seed(0)
        >>> input_batch = {"item_id": torch.LongTensor([[1, 0, 4]])}
        >>> transform = UniformNegativeSamplingTransform(vocab_size=4, num_negative_samples=2)
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'item_id': tensor([[1, 0, 4]]), 'negative_labels': tensor([2, 1])}

    """

    def __init__(
        self,
        vocab_size: int,
        num_negative_samples: int,
        *,
        out_feature_name: Optional[str] = "negative_labels",
        sample_distribution: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        :param vocab_size: The size of sample vocabulary.
        :param num_negative_samples: The size of negatives vector to generate.
        :param out_feature_name: The name of result feature in batch.
        :param sample_distribution: The weightings of indices in the vocabulary. If specified, must
                match the ``vocab_size``. Default: ``None``.
        :param generator: Random number generator to be used for sampling
                from the distribution. Default: ``None``.
        """
        if sample_distribution is not None and sample_distribution.size(-1) != vocab_size:
            msg = (
                "The sample_distribution parameter has an incorrect size. "
                f"Got {sample_distribution.size(-1)}, expected {vocab_size}."
            )
            raise ValueError(msg)

        assert num_negative_samples < vocab_size

        super().__init__()

        self.out_feature_name = out_feature_name
        self.num_negative_samples = num_negative_samples
        self.generator = generator
        if sample_distribution is not None:
            self.sample_distribution = sample_distribution
        else:
            self.sample_distribution = torch.ones(vocab_size)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # [num_negatives] - shape of negatives
        negatives = torch.multinomial(
            self.sample_distribution,
            num_samples=self.num_negative_samples,
            replacement=False,
            generator=self.generator,
        )

        batch[self.out_feature_name] = negatives.to(device=next(iter(batch.values())).device)
        return batch
