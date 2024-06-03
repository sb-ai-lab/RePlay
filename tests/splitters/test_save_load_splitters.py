import pytest

from replay.splitters import (
    ColdUserRandomSplitter,
    KFolds,
    LastNSplitter,
    NewUsersSplitter,
    RandomSplitter,
    RatioSplitter,
    TimeSplitter,
    TwoStageSplitter,
)
from replay.utils.common import load_from_replay, save_to_replay


@pytest.mark.core
@pytest.mark.parametrize(
    "splitter",
    [
        ColdUserRandomSplitter(0.5),
        KFolds(10),
        LastNSplitter(3),
        NewUsersSplitter(0.2),
        RandomSplitter(0.1),
        RatioSplitter(0.13),
        TimeSplitter("2024-02-10 13:20:13"),
        TwoStageSplitter(0.3, 0.5),
    ],
)
def test_equal_attributes(splitter, tmp_path):
    path = (tmp_path / "test").resolve()
    args_names = splitter._init_arg_names
    save_to_replay(splitter, path)
    loaded_splitter = load_from_replay(path)

    for name in args_names:
        assert getattr(splitter, name) == getattr(loaded_splitter, name)
