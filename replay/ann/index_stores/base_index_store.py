from abc import ABC, abstractmethod
from typing import Callable, Any


class IndexStore(ABC):
    """Abstract base class for index stores. Describes a common interface for index stores."""
    def __init__(self):
        self._index = None

    @abstractmethod
    def save_to_store(self, save_index: Callable[[str], None]):
        """Dumps index file to store"""

    @abstractmethod
    def load_index(
        self,
        init_index: Callable[[], None],
        load_index: Callable[[Any, str], None],
        configure_index: Callable[[Any], None],
    ):
        """Loads index from IndexStore to index instance"""

    @abstractmethod
    def dump_index(self, target_path: str):
        """Dumps index files to `target_path`"""
