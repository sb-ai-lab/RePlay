"""
Данный модуль содержит обертки для известных моделей, вроде Lightfm_Wrap,
и реализует некоторые классические алгоритмы.

Модели используют в реализации либо Spark, либо pytorch.
"""

from sponge_bob_magic.models.als import ALSWrap
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.models.base_torch_rec import TorchRecommender
from sponge_bob_magic.models.classifier_rec import ClassifierRec
from sponge_bob_magic.models.implicit_wrap import ImplicitWrap
from sponge_bob_magic.models.knn import KNN
from sponge_bob_magic.models.lightfm_wrap import LightFMWrap
from sponge_bob_magic.models.mult_vae import MultVAE
from sponge_bob_magic.models.neuromf import NeuroMF
from sponge_bob_magic.models.pop_rec import PopRec
from sponge_bob_magic.models.random_rec import RandomRec
from sponge_bob_magic.models.slim import SLIM
from sponge_bob_magic.models.wilson import Wilson
from sponge_bob_magic.models.word2vec import Word2VecRec
