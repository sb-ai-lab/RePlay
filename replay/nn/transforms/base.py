from typing import Any

import torch
from typing_extensions import TypeAlias

FeatureName: TypeAlias = str
FeatureAttrs: TypeAlias = dict[str, Any]


class BaseTransform(torch.nn.Module):
    """
    Base class for all transform modules, defining their required methods.

    A Transform is a Torch module performing batch-wise preprocessing on data fetched via ``ParquetDataset`` instance.
    For using them, a seguence of Transforms should be defined in a ``ParquetModule`` constructor.
    """

    def adjust_meta(self, meta: dict[FeatureName, FeatureAttrs]) -> dict[FeatureName, FeatureAttrs]:
        """
        A method for overriding features' metadata, should the transformation require so.
        If no adjustments are required, this method should return the original meta dict.
        """
        return meta
