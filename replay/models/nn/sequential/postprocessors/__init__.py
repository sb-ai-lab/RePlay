from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from ._base import BasePostProcessor
    from .postprocessors import RemoveSeenItems, SampleItems
