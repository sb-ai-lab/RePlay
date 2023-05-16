from dataclasses import dataclass
from typing import Literal, ClassVar

from replay.ann.entities.base_hnsw_param import BaseHnswParam


@dataclass
class NmslibHnswParam(BaseHnswParam):
    """
    Parameters for nmslib-hnsw methods.

    For example,

    >>> NmslibHnswParam(space='negdotprod_sparse',\
                        M=10,\
                        efS=200,\
                        efC=200,\
                        post=0,\
                        build_index_on='driver'\
        )
    or
    >>> NmslibHnswParam(space='negdotprod_sparse',\
                        M=10,\
                        efS=200,\
                        efC=200,\
                        post=0,\
                        build_index_on='executor'\
                        index_path="/tmp/nmslib_hnsw_index"\
        )

    The reasonable range of values for M parameter is 5-100,
    for efC and eFS is 100-2000.
    Increasing these values improves the prediction quality
    but increases index_time and inference_time too.

     We recommend using these settings:

    - M=16, efC=200 and efS=200 for simple datasets like MovieLens.
    - M=50, efC=1000 and efS=1000 for average quality with an average prediction time.
    - M=75, efC=2000 and efS=2000 for the highest quality with a long prediction time.

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

    method: ClassVar[str] = "hnsw"
