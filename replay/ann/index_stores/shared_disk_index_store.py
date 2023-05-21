import os
from pathlib import Path
from typing import Callable, Any

from replay.ann.index_stores.base_index_store import IndexStore


class SharedDiskIndexStore(IndexStore):
    def __init__(self, warehouse_dir: str, index_dir: str):
        self.index_dir_path = os.path.join(warehouse_dir, index_dir)
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

        temp_file_path = os.path.join(self.index_dir_path, "index")
        load_index(self._index, temp_file_path)
        configure_index(self._index)

        return self._index

    def save_to_store(self, save_index: Callable[[str], None]):
        Path(self.index_dir_path).mkdir(parents=True, exist_ok=True)
        temp_file_path = os.path.join(self.index_dir_path, "index")
        save_index(temp_file_path)
