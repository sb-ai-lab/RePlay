from dataclasses import dataclass, field
from typing import ClassVar, Optional
from typing_extensions import Literal

from replay.models.extensions.ann.entities.base_hnsw_param import BaseHnswParam


@dataclass
class NmslibHnswParam(BaseHnswParam):
    """
    Parameters for nmslib-hnsw methods.

    For example,

    >>> NmslibHnswParam(space='negdotprod_sparse',\
                        m=10,\
                        ef_s=200,\
                        ef_c=200,\
                        post=0,\
        )
    NmslibHnswParam(space='negdotprod_sparse', m=10, ef_c=200, post=0, ef_s=200, items_count=None)

    or

    >>> NmslibHnswParam(space='negdotprod_sparse',\
                        m=10,\
                        ef_s=200,\
                        ef_c=200,\
                        post=0,\
        )
    NmslibHnswParam(space='negdotprod_sparse', m=10, ef_c=200, post=0, ef_s=200, items_count=None)

    The reasonable range of values for `m` parameter is 5-100,
    for `ef_c` and `ef_s` is 100-2000.
    Increasing these values improves the prediction quality
    but increases index_time and inference_time too.

     We recommend using these settings:

    - m=16, ef_c=200 and ef_s=200 for simple datasets like MovieLens.
    - m=50, ef_c=1000 and ef_s=1000 for average quality with an average prediction time.
    - m=75, ef_c=2000 and ef_s=2000 for the highest quality with a long prediction time.

    note: choosing these parameters depends on the dataset
    and quality/time tradeoff.

    note: while reducing parameter values the highest range metrics
    like Metric@1000 suffer first.

    note: even in a case with a long training time,
    profit from ann could be obtained while inference will be used multiple times.

    For more details see https://github.com/nmslib/nmslib/blob/master/manual/methods.md.
    """

    space: Literal[
        "cosinesimil_sparse",
        "cosinesimil_sparse_fast",
        "negdotprod_sparse",
        "negdotprod_sparse_fast",
        "angulardist_sparse",
        "angulardist_sparse_fast",
    ] = "negdotprod_sparse_fast"
    items_count: Optional[int] = field(default=None, init=False)

    method: ClassVar[str] = "hnsw"

    # def init_args_as_dict(self):
    #     # union dicts
    #     return dict(
    #         super().init_args_as_dict()["init_args"], **{"space": self.space}
    #     )
