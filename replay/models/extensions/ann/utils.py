import hnswlib
import nmslib

from .entities.hnswlib_param import HnswlibParam
from .entities.nmslib_hnsw_param import NmslibHnswParam


def create_hnswlib_index_instance(params: HnswlibParam, init: bool = False):
    """
    Creates and returns hnswlib index

    :param params: `HnswlibParam`
    :param init: If `True` it will call the `init_index` method on the index.
        Used when we want to create a new index.
        If `False` then the index will be used to load index data from a file.
    :return: `hnswlib` index instance
    """
    index = hnswlib.Index(space=params.space, dim=params.dim)

    if init:
        # Initializing index - the maximum number of elements should be known beforehand
        index.init_index(
            max_elements=params.max_elements,
            ef_construction=params.ef_c,
            M=params.m,
        )

    return index


def create_nmslib_index_instance(params: NmslibHnswParam):
    """
    Creates and returns nmslib index

    :param params: `NmslibHnswParam`
    :return: `nmslib` index
    """
    index = nmslib.init(
        method=params.method,
        space=params.space,
        data_type=nmslib.DataType.SPARSE_VECTOR,
    )

    return index
