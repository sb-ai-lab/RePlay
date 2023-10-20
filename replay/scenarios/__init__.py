"""
Scenarios are a series of actions for recommendations
"""
from .basescenario import BaseScenario
from .fallback import Fallback

__all__ = [
    "Fallback",
    "BaseScenario",
]
