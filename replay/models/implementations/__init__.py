from .commons import IsSavable
from .pandas import _PopRecPandas
from .polars import _PopRecPolars
from .spark import _PopRecSpark
from .spark.base_rec import (
    _BaseRecommenderSparkImpl,
    _HybridRecommenderSparkImpl,
    _ItemVectorModelSparkImpl,
    _NonPersonalizedRecommenderSparkImpl,
    _QueryRecommenderSparkImpl,
    _RecommenderCommonsSparkImpl,
    _RecommenderSparkImpl,
)

implementations_list = [_PopRecPandas, _PopRecPolars, _PopRecSpark]

__all__ = [
    "implementations_list",
    "_BaseRecommenderSparkImpl",
    "_HybridRecommenderSparkImpl",
    "_ItemVectorModelSparkImpl",
    "_NonPersonalizedRecommenderSparkImpl",
    "_RecommenderCommonsSparkImpl",
    "_RecommenderSparkImpl",
    "IsSavable",
    "_PopRecPandas",
    "_PopRecPolars",
    "_PopRecSpark",
    "_QueryRecommenderSparkImpl",
]
