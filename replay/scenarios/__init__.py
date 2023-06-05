"""
Scenarios are a series of actions for recommendations
"""
from replay.scenarios.fallback import Fallback
from replay.scenarios.basescenario import BaseScenario
from replay.scenarios.one_stage import OneStageScenario, OneStageItem2ItemScenario, OneStageUser2ItemScenario
from replay.scenarios.two_stages.two_stages_scenario import TwoStagesScenario
from replay.scenarios.autorecsys_scenario import AutoRecSysScenario
