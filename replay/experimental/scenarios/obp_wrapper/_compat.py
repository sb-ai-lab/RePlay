import sys

from replay.utils.types import OBP_AVAILABLE, FeatureUnavailableError


def _ensure_obp_available() -> None:
    if not OBP_AVAILABLE:  # pragma: no cover
        err = FeatureUnavailableError("`obp_wrapper` can only be provided when SB-OBP is installed.")
        if sys.version_info >= (3, 13):  # pragma: py-lt-313
            err.add_note("SB-OBP does not support Python >= 3.13")
        raise err
