from typing import Union

import nmslib
from pyspark import SparkFiles

from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.ann.utils import load_index_from_source_fs
from replay.utils import FileInfo


# pylint: disable=too-few-public-methods
class NmslibIndexFileManager:
    """Loads index from hdfs, local disk or SparkFiles dir and keep it in a memory.
    Instance of `NmslibIndexFileManager` broadcasts to executors and is used in pandas_udf.
    """

    def __init__(
        self, index_params: NmslibHnswParam, index_file: Union[FileInfo, str]
    ) -> None:
        self._method = index_params.method
        self._space = index_params.space
        self._ef_s = index_params.efS
        self._index_file = index_file
        self._index = None

    @property
    def index(self):
        """Loads `nmslib hnsw` index from local disk, hdfs or spark files directory and returns it.
        Loads the index only on the first call, then the loaded index is used.

        :return: `nmslib hnsw` index instance
        """
        if self._index:
            return self._index

        self._index = nmslib.init(  # pylint: disable=c-extension-no-member
            method=self._method,
            space=self._space,
            data_type=nmslib.DataType.SPARSE_VECTOR,  # pylint: disable=c-extension-no-member
        )
        if isinstance(self._index_file, FileInfo):
            load_index_from_source_fs(
                sparse=True,
                load_index=lambda path: self._index.loadIndex(
                    path, load_data=True
                ),
                source=self._index_file,
            )
        else:
            self._index.loadIndex(
                SparkFiles.get(self._index_file), load_data=True
            )

        if self._ef_s:
            self._index.setQueryTimeParams({"efSearch": self._ef_s})
        return self._index
