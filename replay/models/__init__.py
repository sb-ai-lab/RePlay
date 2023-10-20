"""
This module contains recommender system algorithms including:

- distributed models built in PySpark
- neural networks build in PyTorch with distributed inference in PySpark
- wrappers for commonly used recommender systems libraries and
    models with non-distributed training and distributed inference in PySpark.
"""

from replay.models.association_rules import AssociationRulesItemRec
from replay.models.base_rec import Recommender
from replay.models.knn import ItemKNN
from replay.models.pop_rec import PopRec
from replay.models.user_pop_rec import UserPopRec
from replay.models.random_rec import RandomRec
from replay.models.slim import SLIM
from replay.models.wilson import Wilson
from replay.models.word2vec import Word2VecRec
from replay.models.cluster import ClusterRec
from replay.models.cat_pop_rec import CatPopRec
from replay.models.ucb import UCB
from replay.models.thompson_sampling import ThompsonSampling
from replay.models.als import ALSWrap
