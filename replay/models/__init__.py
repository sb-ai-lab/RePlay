"""
This module contains recommender system algorithms including:

- distributed models built in PySpark
- neural networks build in PyTorch with distributed inference in PySpark
- wrappers for commonly used recommender systems libraries and\
models with non-distributed training and distributed inference in PySpark.
"""

from .als import ALSWrap
from .association_rules import AssociationRulesItemRec
from .base_rec_client import BaseRecommenderClient, NonPersonolizedRecommenderClient
from .cat_pop_rec import CatPopRec
from .cluster import ClusterRec
from .implementations import (
    _BaseRecommenderSparkImpl,
    _HybridRecommenderSparkImpl,
    _ItemVectorModelSparkImpl,
    _NonPersonalizedRecommenderSparkImpl,
    _RecommenderCommonsSparkImpl,
    _RecommenderSparkImpl,
)
from .kl_ucb import KLUCB
from .knn import ItemKNN
from .lin_ucb import LinUCB
from .pop_rec import PopRec
from .query_pop_rec import QueryPopRec
from .random_rec import RandomRec
from .slim import SLIM
from .thompson_sampling import ThompsonSampling
from .ucb import UCB
from .wilson import Wilson
from .word2vec import Word2VecRec

client_model_list = [PopRec]

__all__ = [
    "ALSWrap",
    "AssociationRulesItemRec",
    "CatPopRec",
    "ClusterRec",
    "_BaseRecommenderSparkImpl",
    "_HybridRecommenderSparkImpl",
    "_ItemVectorModelSparkImpl",
    "_NonPersonalizedRecommenderSparkImpl",
    "_RecommenderCommonsSparkImpl",
    "_RecommenderSparkImpl",
    "BaseRecommenderClient",
    "NonPersonolizedRecommenderClient",
    "KLUCB",
    "ItemKNN",
    "LinUCB",
    "PopRec",
    "QueryPopRec",
    "RandomRec",
    "SLIM",
    "ThompsonSampling",
    "UCB",
    "Wilson",
    "Word2VecRec",
    "client_model_list",
]
