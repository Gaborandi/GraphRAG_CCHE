# utils/error_handler.py
from typing import Optional, Dict, Any, Type
import logging
import traceback
from functools import wraps
from datetime import datetime

class GraphError(Exception):
    """Base class for graph-related errors."""
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()

class ValidationError(GraphError):
    """Error raised for validation failures."""
    pass

class OperationError(GraphError):
    """Error raised for operation failures."""
    pass

class DatabaseError(GraphError):
    """Error raised for database-related failures."""
    pass

class ConfigurationError(GraphError):
    """Error raised for configuration-related failures."""
    pass

def handle_errors(logger: Optional[logging.Logger] = None,
                 raise_error: bool = True,
                 default_value: Any = None):
    """
    Decorator for handling errors in functions.
    Args:
        logger: Logger instance for error logging
        raise_error: Whether to raise the error or return default value
        default_value: Value to return if error occurs and raise_error is False
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except GraphError as e:
                if logger:
                    logger.error(
                        f"GraphError in {func.__name__}: {str(e)}",
                        extra={
                            'error_code': e.error_code,
                            'details': e.details,
                            'timestamp': e.timestamp,
                            'traceback': traceback.format_exc()
                        }
                    )
                if raise_error:
                    raise
                return default_value
            except Exception as e:
                if logger:
                    logger.error(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        extra={
                            'error_code': 'unexpected_error',
                            'details': {
                                'type': type(e).__name__,
                                'args': str(e.args)
                            },
                            'traceback': traceback.format_exc()
                        }
                    )
                if raise_error:
                    raise GraphError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        'unexpected_error',
                        {
                            'original_error': type(e).__name__,
                            'original_message': str(e)
                        }
                    )
                return default_value
        return wrapper
    return decorator

class ErrorTracker:
    """Tracks and analyzes errors for monitoring."""
    
    def __init__(self):
        self.errors = []
        self.error_counts = {}
        
    def track_error(self, error: GraphError):
        """Track an error occurrence."""
        error_info = {
            'error_code': error.error_code,
            'message': str(error),
            'details': error.details,
            'timestamp': error.timestamp,
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_info)
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.errors:
            return {
                'total_errors': 0,
                'error_counts': {},
                'latest_error': None,
                'error_rate': 0.0
            }
            
        latest = self.errors[-1]
        total = len(self.errors)
        
        # Calculate error rate (errors per minute)
        if total >= 2:
            time_span = (self.errors[-1]['timestamp'] - self.errors[0]['timestamp']).total_seconds()
            error_rate = (total / time_span) * 60 if time_span > 0 else 0.0
        else:
            error_rate = 0.0
            
        return {
            'total_errors': total,
            'error_counts': self.error_counts,
            'latest_error': latest,
            'error_rate': error_rate
        }
        
    def clear_errors(self):
        """Clear error history."""
        self.errors = []
        self.error_counts = {}

def format_error_message(error: GraphError) -> str:
    """Format error message with details."""
    message = f"Error: {str(error)} (Code: {error.error_code})"
    
    if error.details:
        message += "\nDetails:"
        for key, value in error.details.items():
            message += f"\n  {key}: {value}"
            
    return message

def error_to_dict(error: GraphError) -> Dict[str, Any]:
    """Convert error to dictionary format."""
    return {
        'message': str(error),
        'error_code': error.error_code,
        'details': error.details,
        'timestamp': error.timestamp.isoformat(),
        'type': type(error).__name__,
        'traceback': traceback.format_exc()
    }

def create_error_response(error: GraphError) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        'success': False,
        'error': error_to_dict(error),
        'timestamp': datetime.now().isoformat()
    }

# Global error tracker instance
error_tracker = ErrorTracker()