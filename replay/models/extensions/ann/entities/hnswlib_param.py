from dataclasses import dataclass, field

from typing_extensions import Literal

from .base_hnsw_param import BaseHnswParam


@dataclass
class HnswlibParam(BaseHnswParam):
    """
    Parameters for hnswlib methods.

    For example,

    >>> HnswlibParam(space="ip",\
                     m=100,\
                     ef_c=200,\
                     post=0,\
                     ef_s=2000,\
        )
    HnswlibParam(space='ip', m=100, ef_c=200, post=0, ef_s=2000, dim=None, max_elements=None)

    or

    >>> HnswlibParam(space="ip",\
                     m=100,\
                     ef_c=200,\
                     post=0,\
                     ef_s=2000,\
        )
    HnswlibParam(space='ip', m=100, ef_c=200, post=0, ef_s=2000, dim=None, max_elements=None)

    The "space" parameter described on the page https://github.com/nmslib/hnswlib/blob/master/README.md#supported-distances
    Parameters "m", "ef_s" and "ef_c" are described at https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

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
    """

    space: Literal["l2", "ip", "cosine"] = "ip"
    # Dimension of vectors in index
    dim: int = field(default=None, init=False)
    # Max number of elements that will be stored in the index
    max_elements: int = field(default=None, init=False)
