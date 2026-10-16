from .base import LossProto
from .bce import BCE, BCESampled
from .ce import CE, CESampled, CESampledWeighted, CEWeighted
from .entmax import PREntmax, PREntmaxSampled
from .grouped_ce import GroupedCESampled
from .grouped_login_ce import GroupedLogInCESampled
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
    "GroupedCESampled",
    "GroupedLogInCESampled",
    "LogInCE",
    "LogInCESampled",
    "LogOutCE",
    "LogOutCESampled",
    "LogOutCEWeighted",
    "LossProto",
    "PREntmax",
    "PREntmaxSampled",
]
