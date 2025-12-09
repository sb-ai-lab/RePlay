from replay.utils import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    msg = (
        "The replay.nn module is unavailable. "
        "To use the functionality from this module, please install ``torch`` and ``lightning``."
    )
    raise ImportError(msg)

from .agg import AggregatorProto, ConcatAggregator, SumAggregator
from .embedding import CategoricalEmbedding, NumericalEmbedding, SequenceEmbedding
from .ffn import PointWiseFeedForward, SwiGLU, SwiGLUEncoder
from .head import EmbeddingTyingHead
from .lightning import LightningModule
from .mask import AttentionMaskProto, DefaultAttentionMask
from .normalization import NormalizerProto
from .output import InferenceOutput, TrainOutput
from .utils import create_activation
