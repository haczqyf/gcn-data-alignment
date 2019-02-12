"""Definition of custom Exception types"""


class IngredientsException(BaseException):
    """Base exception for this project."""

    pass


class NormTypeNotPossible(IngredientsException):
    """Exception raised if no norm ty is possible"""

    pass