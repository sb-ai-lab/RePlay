"""
This module contains Learner class for RePlay models training on bandit dataset.
The format of the bandit dataset should be the same as in OpenBanditPipeline.
Learner class has methods `fit` and `predict` which are wrappers for the corresponding
methods of RePlay model. Optimize is based on optimization over CTR estimated by OBP.
"""

import sys

from replay.utils.types import OBP_AVAILABLE, FeatureUnavailableError

if OBP_AVAILABLE:
    from replay.experimental.scenarios.obp_wrapper.replay_offline import OBPOfflinePolicyLearner
else:  # pragma: no cover

    class OBPOfflinePolicyLearner:  # type: ignore[no-redef]
        """Fallback class raising a clear error when SB-OBP is missing."""

        def __init__(self, *args, **kwargs):
            del args, kwargs
            err = FeatureUnavailableError("`obp_wrapper` can only be provided when SB-OBP is installed.")
            if sys.version_info >= (3, 13):  # pragma: py-lt-313
                err.add_note("SB-OBP does not support Python >= 3.13")
            raise err
