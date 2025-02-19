"""
This module contains tools for preprocessing data including:

- filters
- processors for feature transforms
"""

from .converter import CSRConverter
from .discretizer import (
    Discretizer,
    GreedyDiscretizingRule,
    HandleInvalidStrategies,
    QuantileDiscretizingRule,
)
from .history_based_fp import (
    ConditionalPopularityProcessor,
    EmptyFeatureProcessor,
    HistoryBasedFeaturesProcessor,
    LogStatFeaturesProcessor,
)
from .label_encoder import LabelEncoder, LabelEncoderPartialFitWarning, LabelEncodingRule, SequenceEncodingRule
from .sessionizer import Sessionizer
