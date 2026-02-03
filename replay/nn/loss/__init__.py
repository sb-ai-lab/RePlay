from .base import LossProto
from .bce import BCE, BCESampled
from .ce import CE, CESampled, CESampledWeighted, CEWeighted
from .login_ce import LogInCE, LogInCESampled
from .logout_ce import LogOutCE, LogOutCEWeighted
from .composed import ComposedLoss

LogOutCESampled = CE

__all__ = [
    "BCE",
    "CE",
    "BCESampled",
    "ComposedLoss",
    "CESampled",
    "CESampledWeighted",
    "CEWeighted",
    "LogInCE",
    "LogInCESampled",
    "LogOutCE",
    "LogOutCESampled",
    "LogOutCEWeighted",
    "LossProto",
]
