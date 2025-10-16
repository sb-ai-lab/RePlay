from typing import Any

from replay.experimental.models.admm_slim import ADMMSLIM
from replay.experimental.models.base_torch_rec import TorchRecommender
from replay.experimental.models.cql import CQL
from replay.experimental.models.ddpg import DDPG
from replay.experimental.models.dt4rec.dt4rec import DT4Rec
from replay.experimental.models.hierarchical_recommender import HierarchicalRecommender
from replay.experimental.models.implicit_wrap import ImplicitWrap
from replay.experimental.models.mult_vae import MultVAE
from replay.experimental.models.neural_ts import NeuralTS
from replay.experimental.models.neuromf import NeuroMF
from replay.experimental.models.scala_als import ScalaALSWrap
from replay.experimental.models.u_lin_ucb import ULinUCB

__all__ = [
    "ADMMSLIM",
    "CQL",
    "DDPG",
    "DT4Rec",
    "HierarchicalRecommender",
    "ImplicitWrap",
    "MultVAE",
    "NeuralTS",
    "NeuroMF",
    "ScalaALSWrap",
    "TorchRecommender",
    "ULinUCB",
]

CONDITIONAL_IMPORTS = {"LightFMWrap": "replay.experimental.models.lightfm_wrap"}


class ConditionalAccessError(Exception):
    """Raised when trying to access conditional elements from parent module instead of a direct import."""


def __getattr__(name: str) -> Any:
    if name in CONDITIONAL_IMPORTS:
        msg = (
            f"{name} relies on manual dependency installation and cannot be accessed via higher-level modules. "
            f"If you wish to use this attribute, import it directly from {CONDITIONAL_IMPORTS[name]}"
        )

        raise ConditionalAccessError(msg)

    if name in __all__:
        return globals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
