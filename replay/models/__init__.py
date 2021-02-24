"""
Данный модуль содержит обертки для известных моделей, вроде Lightfm_Wrap,
и реализует некоторые классические алгоритмы.

Модели используют в реализации либо Spark, либо pytorch.
"""

from replay.models.admm_slim import ADMMSLIM
from replay.models.als import ALSWrap
from replay.models.base_rec import Recommender
from replay.models.base_torch_rec import TorchRecommender
from replay.models.classifier_rec import ClassifierRec
from replay.models.implicit_wrap import ImplicitWrap
from replay.models.knn import KNN
from replay.models.lightfm_wrap import LightFMWrap
from replay.models.mult_vae import MultVAE
from replay.models.neuromf import NeuroMF
from replay.models.pop_rec import PopRec
from replay.models.user_pop_rec import UserPopRec
from replay.models.random_rec import RandomRec
from replay.models.slim import SLIM
from replay.models.wilson import Wilson
from replay.models.word2vec import Word2VecRec
from replay.models.stack import Stack
from replay.models.cluster import ColdUser
