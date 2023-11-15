import logging
import os
import tempfile
import weakref
from typing import Any, Callable

from pyarrow import fs

from replay.models.extensions.ann.index_stores.base_index_store import IndexStore
from replay.models.extensions.ann.index_stores.utils import FileSystem, get_filesystem

logger = logging.getLogger("replay")


class HdfsIndexStore(IndexStore):
    """Class that responsible for index store in HDFS."""

    def __init__(self, warehouse_dir: str, index_dir: str):
        index_dir_path = os.path.join(warehouse_dir, index_dir)
        self._index_dir_info = get_filesystem(index_dir_path)
        if self._index_dir_info.filesystem != FileSystem.HDFS:
            raise ValueError(
                f"Can't recognize path {index_dir_path} as HDFS path!"
            )
        self._hadoop_fs = fs.HadoopFileSystem.from_uri(
            self._index_dir_info.hdfs_uri
        )
        super().__init__()

        if self.cleanup:
            logger.debug(
                "Index directory %s is marked for deletion via weakref.finalize()",
                self._index_dir_info.path,
            )
            weakref.finalize(
                self, self._hadoop_fs.delete_dir, self._index_dir_info.path
            )

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
                self._index_dir_info.path,
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
                self._index_dir_info.path,
                destination_filesystem=self._hadoop_fs,
            )
            # param use_threads=True (?)
            logger.info("Index files saved to %s", self._index_dir_info.path)

    def dump_index(self, target_path: str):
        target_path_info = get_filesystem(target_path)
        destination_filesystem, target_path = fs.FileSystem.from_uri(
            target_path_info.hdfs_uri + target_path_info.path
            if target_path_info.filesystem == FileSystem.HDFS
            else target_path_info.path
        )
        target_path = os.path.join(target_path, "index_files")
        fs.copy_files(
            self._index_dir_info.path,
            target_path,
            source_filesystem=self._hadoop_fs,
            destination_filesystem=destination_filesystem,
        )
