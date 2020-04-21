"""
Данный модуль содержит обертки для известных моделей, вроде LightFM,
и реализует некоторые классические алгоритмы.

Модели используют в реализации либо Spark, либо pytorch.
"""

from sponge_bob_magic.models.als import ALSWrap
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.models.classifier_rec import ClassifierRec
from sponge_bob_magic.models.knn import KNN
from sponge_bob_magic.models.lightfm import LightFMWrap
from sponge_bob_magic.models.mult_vae import MultVAE
from sponge_bob_magic.models.neuromf import NeuroMF
from sponge_bob_magic.models.pop_rec import PopRec
from sponge_bob_magic.models.random_rec import RandomRec
from sponge_bob_magic.models.slim import SLIM
