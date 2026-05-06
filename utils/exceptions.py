"""Domain-specific exceptions used across the forecasting system."""


class ForecastingError(Exception):
    """Base exception for expected forecasting failures."""


class DataValidationError(ForecastingError):
    """Raised when the input dataset is missing required columns or values."""


class ModelNotFoundError(ForecastingError):
    """Raised when a requested state model artifact is unavailable."""
