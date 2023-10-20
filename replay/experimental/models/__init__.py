from .admm_slim import ADMMSLIM
from .base_torch_rec import TorchRecommender
from .cql import CQL
from .ddpg import DDPG
from .implicit_wrap import ImplicitWrap
from .lightfm_wrap import LightFMWrap
from .mult_vae import MultVAE
from .neuromf import NeuroMF
from .scala_als import ScalaALSWrap

__all__ = [
    "extensions",
    "ADMMSLIM",
    "ScalaALSWrap",
    "ImplicitWrap",
    "LightFMWrap",
    "MultVAE",
    "NeuroMF",
    "CQL",
    "DDPG",
    "TorchRecommender",
]
