import os
import tempfile
from typing import Callable, Any

from pyarrow import fs
from pyspark import SparkFiles

from replay.ann.index_stores.base_index_store import IndexStore
from replay.session_handler import State
from replay.utils import get_filesystem, FileSystem


class SparkFilesIndexStore(IndexStore):
    """Class that responsible for index store in spark files.
    Works though SparkContext.addFile()."""
    def __init__(self):
        self.index_dir_path = None
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

        temp_file_path = SparkFiles.get("index")
        load_index(self._index, temp_file_path)
        configure_index(self._index)

        return self._index

    def save_to_store(self, save_index: Callable[[str], None]):
        self.index_dir_path: str = tempfile.mkdtemp()
        temp_file_path = os.path.join(self.index_dir_path, "index")
        save_index(temp_file_path)

        spark = State().session
        for filename in os.listdir(self.index_dir_path):
            spark.sparkContext.addFile(
                "file://" + os.path.join(self.index_dir_path, filename)
            )

    def dump_index(self, target_path: str):
        destination_filesystem, target_path = fs.FileSystem.from_uri(
            target_path
        )
        target_path = os.path.join(target_path, "index_files")
        destination_filesystem.create_dir(target_path)
        fs.copy_files(
            self.index_dir_path,
            target_path,
            source_filesystem=fs.LocalFileSystem(),
            destination_filesystem=destination_filesystem,
        )

    def load_from_path(self, path: str):
        """Loads index from `path` directory to spark files."""
        path_info = get_filesystem(path)
        source_filesystem, path = fs.FileSystem.from_uri(
            path_info.hdfs_uri + path_info.path
            if path_info.filesystem == FileSystem.HDFS
            else path_info.path
        )
        path = os.path.join(path, "index_files")
        self.index_dir_path: str = tempfile.mkdtemp()
        fs.copy_files(
            path,
            self.index_dir_path,
            source_filesystem=source_filesystem,
            destination_filesystem=fs.LocalFileSystem(),
        )

        spark = State().session
        for filename in os.listdir(self.index_dir_path):
            spark.sparkContext.addFile(
                "file://" + os.path.join(self.index_dir_path, filename)
            )
