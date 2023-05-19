from dataclasses import dataclass
from typing import Optional
from typing_extensions import Literal


@dataclass
class BaseHnswParam:
    """
    Base hnsw params.
    """

    M: int = 200
    efC: int = 20000
    post: int = 0
    efS: Optional[int] = None
    build_index_on: Literal["driver", "executor"] = "driver"
    index_path: Optional[str] = None

    def __post_init__(self):
        if self.build_index_on == "executor":
            assert (
                self.index_path
            ), 'if build_index_on == "executor" then index_path must be set!'

    def init_params_as_dict(self):
        return {
            "M": self.M,
            "efC": self.efC,
            "post": self.post,
            "efS": self.efS,
        }
