from dataclasses import dataclass, field
from typing_extensions import Literal

from replay.ann.entities.base_hnsw_param import BaseHnswParam


@dataclass
class HnswlibParam(BaseHnswParam):
    """
    Parameters for hnswlib methods.

    For example,

    >>> HnswlibParam(space="ip",\
                     M=100,\
                     efC=200,\
                     post=0,\
                     efS=2000,\
                     build_index_on="driver"\
        )
    HnswlibParam(M=100, efC=200, post=0, efS=2000, build_index_on='driver', index_path=None, space='ip', dim=None, max_elements=None)

    or

    >>> HnswlibParam(space="ip",\
                     M=100,\
                     efC=200,\
                     post=0,\
                     efS=2000,\
                     build_index_on="executor",\
                     index_path="/tmp/hnswlib_index"\
        )
    HnswlibParam(M=100, efC=200, post=0, efS=2000, build_index_on='executor', index_path='/tmp/hnswlib_index', space='ip', dim=None, max_elements=None)

    The "space" parameter described on the page https://github.com/nmslib/hnswlib/blob/master/README.md#supported-distances
    Parameters "M", "efS" and "efC" are described at https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

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
    """

    space: Literal["l2", "ip", "cosine"] = "ip"
    # Dimension of vectors in index
    dim: int = field(default=None, init=False)
    # Max number of elements that will be stored in the index
    max_elements: int = field(default=None, init=False)

    def init_params_as_dict(self):
        # union dicts
        return dict(super().init_params_as_dict(), **{"space": self.space})
