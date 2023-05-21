import logging
from abc import ABC
from typing import Optional

from replay.ann.ann_mixin import ANNMixin
from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam

logger = logging.getLogger("replay")


class NmslibHnswMixin(ANNMixin, ABC):
    """Mixin that provides methods to build nmslib hnsw index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    _nmslib_hnsw_params: Optional[NmslibHnswParam] = None
    INDEX_FILENAME = "nmslib_hnsw_index"

    def _save_nmslib_hnsw_index(self, path):
        """Method save (copy) index from hdfs (or local) to `path` directory.
        `path` can be an hdfs path or a local path.

        Args:
            path (_type_): directory where to dump (copy) the index
        """

        self._save_index_files(path, self._nmslib_hnsw_params, (".dat",))

    def _load_nmslib_hnsw_index(self, path: str):
        """Loads hnsw index from `path` directory to local dir.
        Index file name is 'hnswlib_index'.
        And adds index file to the `SparkFiles`.
        `path` can be a hdfs path or a local path.


        Args:
            path: directory path, where index file is stored
        """

        print(self._nmslib_hnsw_params)
        self._load_index(path, self._nmslib_hnsw_params, (".dat",))
        print(self._nmslib_hnsw_params)
