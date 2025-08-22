"""
This module contains Learner class for RePlay models training on bandit dataset.
The format of the bandit dataset should be the same as in OpenBanditPipeline.
Learner class has methods `fit` and `predict` which are wrappers for the corresponding
methods of RePlay model. Optimize is based on optimization over CTR estimated by OBP.
"""

from replay.experimental.scenarios.obp_wrapper.replay_offline import OBPOfflinePolicyLearner
