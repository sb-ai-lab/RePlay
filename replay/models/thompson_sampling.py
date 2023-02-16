import math

from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.metrics import Metric, NDCG
from replay.models.base_rec import NonPersonalizedRecommender


class Thompson(NonPersonalizedRecommender):
    pass
