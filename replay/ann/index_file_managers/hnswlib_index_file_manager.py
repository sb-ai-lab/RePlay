from typing import Union
from pyspark import SparkFiles

import hnswlib

from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.ann.utils import load_index_from_source_fs
from replay.utils import FileInfo


# pylint: disable=too-few-public-methods
class HnswlibIndexFileManager:
    """Loads index from hdfs, local disk or SparkFiles dir and keep it in a memory.
    Instance of `HnswlibIndexFileManager` broadcasts to executors and is used in pandas_udf.
    """

    def __init__(
        self, index_params: HnswlibParam, index_file: Union[FileInfo, str]
    ) -> None:
        self._space = index_params.space
        self._ef_s = index_params.efS
        self._dim = index_params.dim
        self._index_file = index_file
        self._index = None

    @property
    def index(self):
        """Loads `hnswlib` index from local disk, hdfs or spark files directory and returns it.
        Loads the index only on the first call, then the loaded index is used.

        :return: `hnswlib` index
        """
        if self._index:
            return self._index

        self._index = hnswlib.Index(  # pylint: disable=c-extension-no-member
            space=self._space, dim=self._dim
        )
        if isinstance(self._index_file, FileInfo):
            load_index_from_source_fs(
                sparse=False,
                load_index=lambda path: self._index.load_index(
                    path
                ),  # pylint: disable=unnecessary-lambda
                source=self._index_file,
            )
        else:
            self._index.load_index(SparkFiles.get(self._index_file))

        if self._ef_s:
            self._index.set_ef(self._ef_s)
        return self._index
