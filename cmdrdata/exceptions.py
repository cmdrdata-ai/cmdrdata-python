"""
Exception classes for CmdrData Universal SDK
"""


class CMDRDataError(Exception):
    """Base exception for all CmdrData errors"""
    pass


class ConfigurationError(CMDRDataError):
    """Raised when configuration is invalid"""
    pass


class ValidationError(CMDRDataError):
    """Raised when input validation fails"""
    pass


class NetworkError(CMDRDataError):
    """Raised when network operations fail"""
    pass


class TrackingError(CMDRDataError):
    """Raised when usage tracking fails"""
    pass