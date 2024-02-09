import logging
import os
import shutil
import weakref
from pathlib import Path
from typing import Any, Callable

from pyarrow import fs

from .base_index_store import IndexStore

logger = logging.getLogger("replay")


class SharedDiskIndexStore(IndexStore):
    """Class that responsible for index store in shared disk.
    It can also be used with a local disk when the driver and executors
    are running on the same machine."""

    def __init__(
        self, warehouse_dir: str, index_dir: str, cleanup: bool = True
    ):
        self.index_dir_path = os.path.join(warehouse_dir, index_dir)
        super().__init__(cleanup)
        if self.cleanup:
            logger.debug(
                "Index directory %s is marked for deletion via weakref.finalize()",
                self.index_dir_path,
            )
            weakref.finalize(self, shutil.rmtree, self.index_dir_path)

    def load_index(
        self,
        init_index: Callable[[], None],
        load_index: Callable[[Any, str], None],
        configure_index: Callable[[Any], None],
    ):  # pragma: no cover
        if self._index:
            return self._index

        self._index = init_index()

        temp_file_path = os.path.join(self.index_dir_path, "index")
        load_index(self._index, temp_file_path)
        configure_index(self._index)

        return self._index

    def save_to_store(self, save_index: Callable[[str], None]):
        Path(self.index_dir_path).mkdir(parents=True, exist_ok=True)
        temp_file_path = os.path.join(self.index_dir_path, "index")
        save_index(temp_file_path)

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
