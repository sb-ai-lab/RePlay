from typing import Optional

import torch

from replay.utils.session_handler import Borg, get_spark_session, logger_with_settings
from replay.utils.types import PYSPARK_AVAILABLE, MissingImportType

if PYSPARK_AVAILABLE:
    from pyspark.sql import SparkSession
else:
    SparkSession = MissingImportType


# pylint: disable=too-few-public-methods
class State(Borg):
    """
    All modules look for Spark session via this class. You can put your own session here.

    Other parameters are stored here too: ``default device`` for ``pytorch`` (CPU/CUDA)
    """

    def __init__(
        self,
        session: Optional[SparkSession] = None,
        device: Optional[torch.device] = None,
    ):
        Borg.__init__(self)
        if not hasattr(self, "logger_set"):
            self.logger = logger_with_settings()
            self.logger_set = True

        if session is None:
            if not hasattr(self, "session"):
                self.session = get_spark_session()
        else:
            self.session = session

        if device is None:
            if not hasattr(self, "device"):
                if torch.cuda.is_available():
                    self.device = torch.device(
                        f"cuda:{torch.cuda.current_device()}"
                    )
                else:
                    self.device = torch.device("cpu")
        else:
            self.device = device
