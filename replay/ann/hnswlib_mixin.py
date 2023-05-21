import logging
from abc import ABC
from typing import Optional

from replay.ann.ann_mixin import ANNMixin
from replay.ann.entities.hnswlib_param import HnswlibParam

logger = logging.getLogger("replay")


class HnswlibMixin(ANNMixin, ABC):
    """Mixin that provides methods to build hnswlib index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    _hnswlib_params: Optional[HnswlibParam] = None
    INDEX_FILENAME = "hnswlib_index"

    def _save_hnswlib_index(self, path: str):
        """Method save (copy) index from hdfs (or local) to `path` directory.
        `path` can be a hdfs path or a local path.

        Args:
            path (str): directory where to dump (copy) the index
        """

        self._save_index_files(path, self._hnswlib_params)

    def _load_hnswlib_index(self, path: str):
        """Loads hnsw index from `path` directory to local dir.
        Index file name is 'hnswlib_index'.
        And adds index file to the `SparkFiles`.
        `path` can be an hdfs path or a local path.


        Args:
            path: directory path, where index file is stored
        """

        self._load_index(path, self._hnswlib_params)
