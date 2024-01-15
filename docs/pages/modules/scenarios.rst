Scenarios
==========
.. automodule:: replay.scenarios

Fallback
---------
.. autoclass:: replay.scenarios.Fallback
   :special-members: __init__
   :members: optimize

Two Stage Scenario (Experimental)
--------------------------------------
.. autoclass:: replay.experimental.scenarios.TwoStagesScenario
   :special-members: __init__
   :members: fit, predict, optimize

Offline Policy Learners
-----------------------
.. autoclass:: replay.experimental.scenarios.obp_wrapper.OBPOfflinePolicyLearner
   :special-members: __init__
   :members: predict, optimize
