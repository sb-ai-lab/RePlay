from .base import LossInfo, LossOutput, LossProto
from .bce import BCE, BCESampled
from .ce import CE, CESampled, CESampledWeighted, CEWeighted
from .composed import ComposedLoss
from .login_ce import LogInCE, LogInCESampled
from .logout_ce import LogOutCE, LogOutCEWeighted

LogOutCESampled = CE

__all__ = [
    "BCE",
    "CE",
    "BCESampled",
    "CESampled",
    "CESampledWeighted",
    "CEWeighted",
    "ComposedLoss",
    "LogInCE",
    "LogInCESampled",
    "LogOutCE",
    "LogOutCESampled",
    "LogOutCEWeighted",
    "LossInfo",
    "LossOutput",
    "LossProto",
]
