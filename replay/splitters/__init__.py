"""
Splits data into train and test
"""

from replay.splitters.base_splitter import (
    Splitter,
    SplitterReturnType,
)
from replay.splitters.log_splitter import (
    NewUsersSplitter,
    ColdUserRandomSplitter,
    DateSplitter,
    RandomSplitter,
)
from replay.splitters.user_log_splitter import UserSplitter, k_folds
