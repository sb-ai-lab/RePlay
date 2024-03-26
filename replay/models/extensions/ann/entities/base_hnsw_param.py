from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseHnswParam:
    """
    Base hnsw params.
    """

    space: str
    m: int = 200
    ef_c: int = 20000
    post: int = 0
    ef_s: Optional[int] = None

    def init_meta_as_dict(self) -> dict:
        """
        Returns meta-information for class instance initialization. Used to save the entity to disk.
        :return: dictionary with init meta.
        """
        return {
            "module": type(self).__module__,
            "class": type(self).__name__,
            "init_args": {
                "space": self.space,
                "m": self.m,
                "ef_c": self.ef_c,
                "post": self.post,
                "ef_s": self.ef_s,
            },
        }
