"""
Splits data into train and test
"""

from replay.splitters.base_splitter import (
    Splitter,
    SplitterReturnType,
)
from replay.splitters.new_users_splitter import NewUsersSplitter
from replay.splitters.cold_user_random_splitter import ColdUserRandomSplitter
from replay.splitters.random_splitter import RandomSplitter
from replay.splitters.two_stage_splitter import TwoStageSplitter
from replay.splitters.ratio_splitter import RatioSplitter
from replay.splitters.last_n_splitter import LastNSplitter
from replay.splitters.time_splitter import TimeSplitter
from replay.splitters.k_folds import KFolds
