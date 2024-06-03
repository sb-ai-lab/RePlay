import json
from pathlib import Path
from typing import Union

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
from replay.utils import TORCH_AVAILABLE

SavableObject = Union[
    ColdUserRandomSplitter,
    KFolds,
    LastNSplitter,
    NewUsersSplitter,
    RandomSplitter,
    RatioSplitter,
    TimeSplitter,
    TwoStageSplitter,
]

if TORCH_AVAILABLE:
    from replay.data.nn import SequenceTokenizer

    SavableObject = Union[
        ColdUserRandomSplitter,
        KFolds,
        LastNSplitter,
        NewUsersSplitter,
        RandomSplitter,
        RatioSplitter,
        TimeSplitter,
        TwoStageSplitter,
        SequenceTokenizer,
    ]


def save_to_replay(obj: SavableObject, path: Union[str, Path]) -> None:
    """
    General function to save RePlay models, splitters and tokenizer.

    :param path: Path to save the object.
    """
    obj.save(path)


def load_from_replay(path: Union[str, Path]) -> SavableObject:
    """
    General function to load RePlay models, splitters and tokenizer.

    :param path: Path to save the object.
    """
    path = Path(path).with_suffix(".replay").resolve()
    with open(path / "init_args.json", "r") as file:
        class_name = json.loads(file.read())["_class_name"]
    obj_type = globals()[class_name]
    obj = obj_type.load(path)

    return obj
