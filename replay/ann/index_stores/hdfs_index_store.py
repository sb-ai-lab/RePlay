import os
import tempfile
from typing import Callable, Any

from pyarrow import fs

from replay.ann.index_stores.base_index_store import IndexStore
from replay.utils import get_filesystem, FileSystem


class HdfsIndexStore(IndexStore):
    def __init__(self, warehouse_dir: str, index_dir: str):
        self._index_dir = get_filesystem(
            os.path.join(warehouse_dir, index_dir)
        )
        if self._index_dir.filesystem != FileSystem.HDFS:
            raise ValueError(
                "Can't recognize path '%s' as HDFS path!", self._index_dir
            )
        self._hadoop_fs = fs.HadoopFileSystem.from_uri(
            self._index_dir.hdfs_uri
        )
        super().__init__()

    def load_index(
        self,
        init_index: Callable[[], None],
        load_index: Callable[[Any, str], None],
        configure_index: Callable[[Any], None],
    ):
        if self._index:
            return self._index

        self._index = init_index()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "index")

            # here we copy index files from hdfs directory
            # to local disk directory
            fs.copy_files(
                self._index_dir.path,
                "file://" + temp_dir,
                source_filesystem=self._hadoop_fs,
            )
            load_index(self._index, temp_file_path)
            configure_index(self._index)

        return self._index

    def save_to_store(self, save_index: Callable[[str], None]):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "index")
            save_index(temp_file_path)

            # here we copy index files from local disk directory
            # to hdfs directory
            fs.copy_files(
                "file://" + temp_dir,
                self._index_dir.path,
                destination_filesystem=self._hadoop_fs,
            )
            # param use_threads=True (?)
