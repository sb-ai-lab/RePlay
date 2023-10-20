"""
This module contains recommender system algorithms including:

- distributed models built in PySpark
- neural networks build in PyTorch with distributed inference in PySpark
- wrappers for commonly used recommender systems libraries and
    models with non-distributed training and distributed inference in PySpark.
"""

from .als import ALSWrap
from .association_rules import AssociationRulesItemRec
from .cat_pop_rec import CatPopRec
from .cluster import ClusterRec
from .knn import ItemKNN
from .pop_rec import PopRec
from .random_rec import RandomRec
from .slim import SLIM
from .thompson_sampling import ThompsonSampling
from .ucb import UCB
from .user_pop_rec import UserPopRec
from .wilson import Wilson
from .word2vec import Word2VecRec

__all__ = [
    "AssociationRulesItemRec",
    "ItemKNN",
    "PopRec",
    "UserPopRec",
    "RandomRec",
    "SLIM",
    "Wilson",
    "Word2VecRec",
    "ClusterRec",
    "CatPopRec",
    "UCB",
    "ThompsonSampling",
    "ALSWrap",
]
