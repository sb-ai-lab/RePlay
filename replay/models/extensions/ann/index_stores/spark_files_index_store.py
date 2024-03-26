import logging
import os
import shutil
import tempfile
import weakref
from typing import Any, Callable

from pyarrow import fs

from replay.utils import PYSPARK_AVAILABLE
from replay.utils.session_handler import State

from .base_index_store import IndexStore
from .utils import FileSystem, get_filesystem

if PYSPARK_AVAILABLE:
    from pyspark import SparkFiles


logger = logging.getLogger("replay")


if PYSPARK_AVAILABLE:

    class SparkFilesIndexStore(IndexStore):
        """Class that responsible for index store in spark files.
        Works through SparkContext.addFile()."""

        def _clean_up(self):  # pragma: no cover
            """Removes directory with index files
            before the instance is garbage collected."""
            if self.index_dir_path:
                shutil.rmtree(self.index_dir_path)

        def __init__(self, cleanup: bool = True):
            self.index_dir_path = None
            super().__init__(cleanup)
            if self.cleanup:
                weakref.finalize(self, self._clean_up)

        def load_index(
            self,
            init_index: Callable[[], None],
            load_index: Callable[[Any, str], None],
            configure_index: Callable[[Any], None],
        ):  # pragma: no cover
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
                index_file_path = os.path.join(self.index_dir_path, filename)
                spark.sparkContext.addFile("file://" + index_file_path)
                logger.info("Index file %s transferred to executors", index_file_path)

        def dump_index(self, target_path: str):
            destination_filesystem, target_path = fs.FileSystem.from_uri(target_path)
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
                path_info.hdfs_uri + path_info.path if path_info.filesystem == FileSystem.HDFS else path_info.path
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
                index_file_path = os.path.join(self.index_dir_path, filename)
                spark.sparkContext.addFile("file://" + index_file_path)
                logger.info("Index file %s transferred to executors", index_file_path)
