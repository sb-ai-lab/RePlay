from typing import Callable

from replay.data.nn.parquet.collate import general_collate
from replay.data.nn.parquet.constants.batches import GeneralCollateFn
from replay.data.nn.parquet.info.replicas import ReplicasInfo, ReplicasInfoProtocol

DEFAULT_COLLATE_FN: GeneralCollateFn = general_collate

DEFAULT_MASK_POSTFIX: str = "_mask"


def default_make_mask_name(postfix: str) -> Callable[[str], str]:
    def function(name: str) -> str:
        return f"{name}{postfix}"

    return function


DEFAULT_MAKE_MASK_NAME = default_make_mask_name(DEFAULT_MASK_POSTFIX)
DEFAULT_REPLICAS_INFO: ReplicasInfoProtocol = ReplicasInfo()
