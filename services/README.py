# services/error_handling.py
from typing import Dict, Any, Optional, Type, Callable, List
import logging
from datetime import datetime, timedelta
import threading
from functools import wraps
import traceback
import json
from enum import Enum
import time
from collections import defaultdict, deque
import asyncio
from dataclasses import dataclass

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    reset_timeout: float = 60.0
    half_open_timeout: float = 30.0
    window_size: int = 10

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    DATABASE = "database"
    PROCESSING = "processing"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    details: Dict[str, Any]
    stack_trace: str

@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1

class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()
        
        # Failure window tracking
        self.failure_window = deque(maxlen=config.window_size)
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': defaultdict(int)
        }

    def __call__(self, func):
        """Decorator for circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self._lock:
                self.stats['total_calls'] += 1
                
                if not self._can_execute():
                    self.stats['rejected_calls'] += 1
                    raise CircuitBreakerError(
                        f"Circuit {self.name} is {self.state.value}"
                    )
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                        
                    self._handle_success()
                    self.stats['successful_calls'] += 1
                    return result
                    
                except Exception as e:
                    self._handle_failure()
                    self.stats['failed_calls'] += 1
                    raise
                    
        return wrapper

    def _can_execute(self) -> bool:
        """Check if execution is allowed."""
        now = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if enough time has passed to try half-open
            if (now - self.last_state_change).total_seconds() >= self.config.reset_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
            
        else:  # HALF_OPEN
            # Only allow one test request
            return True

    def _handle_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.CLOSED)
        
        # Clear failure tracking in closed state
        if self.state == CircuitState.CLOSED:
            self.failure_count = 0
            self.failure_window.clear()

    def _handle_failure(self):
        """Handle execution failure."""
        now = datetime.now()
        self.last_failure_time = now
        self.failure_window.append(now)
        
        # Count failures in the window
        window_start = now - timedelta(seconds=self.config.window_size)
        recent_failures = sum(
            1 for t in self.failure_window
            if t >= window_start
        )
        
        if recent_failures >= self.config.failure_threshold:
            self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.last_state_change = datetime.now()
            
            # Track state change
            self.stats['state_changes'][(old_state, new_state)] += 1
            
            # Log state change
            logging.info(
                f"Circuit {self.name} state change: {old_state.value} -> {new_state.value}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'current_failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time,
                'stats': dict(self.stats)
            }

class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.stats = defaultdict(int)
        self._lock = threading.Lock()

    def __call__(self, func):
        """Decorator for retry logic."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            
            while attempt < self.config.max_attempts:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                        
                    # Track successful retry
                    if attempt > 0:
                        with self._lock:
                            self.stats['successful_retries'] += 1
                            
                    return result
                    
                except Exception as e:
                    attempt += 1
                    last_exception = e
                    
                    with self._lock:
                        self.stats['failed_attempts'] += 1
                    
                    if attempt < self.config.max_attempts:
                        # Calculate delay with exponential backoff
                        delay = min(
                            self.config.base_delay * (
                                self.config.exponential_base ** attempt
                            ),
                            self.config.max_delay
                        )
                        
                        # Add jitter
                        delay *= (1 + self.config.jitter * (random.random() - 0.5))
                        
                        await asyncio.sleep(delay)
                    else:
                        break
            
            # Track max retries exceeded
            with self._lock:
                self.stats['max_retries_exceeded'] += 1
            
            raise MaxRetriesExceededError(
                f"Max retries ({self.config.max_attempts}) exceeded",
                last_exception
            ) from last_exception
            
        return wrapper

    def get_stats(self) -> Dict[str, int]:
        """Get retry statistics."""
        with self._lock:
            return dict(self.stats)

class ErrorManager:
    """Manages error tracking and recovery."""
    
    def __init__(self):
        self.error_history = defaultdict(list)
        self.error_counts = defaultdict(int)
        self._lock = threading.Lock()
        
        # Error handlers by category
        self.error_handlers = {
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.DATABASE: self._handle_database_error,
            ErrorCategory.PROCESSING: self._handle_processing_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.SYSTEM: self._handle_system_error
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorCategory.DATABASE: self._database_recovery_strategy,
            ErrorCategory.NETWORK: self._network_recovery_strategy
        }

    def track_error(self, error: Exception, context: ErrorContext):
        """Track and handle error."""
        with self._lock:
            # Record error
            self.error_history[context.category].append({
                'error': error,
                'context': context,
                'handled_at': datetime.now()
            })
            
            self.error_counts[context.category] += 1
            
            # Handle error
            handler = self.error_handlers.get(
                context.category,
                self._handle_unknown_error
            )
            handler(error, context)
            
            # Attempt recovery if strategy exists
            recovery_strategy = self.recovery_strategies.get(context.category)
            if recovery_strategy:
                recovery_strategy(context)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error tracking summary."""
        with self._lock:
            summary = {
                'total_errors': sum(self.error_counts.values()),
                'errors_by_category': dict(self.error_counts),
                'recent_errors': self._get_recent_errors(),
                'error_trends': self._calculate_error_trends()
            }
            return summary

    def _get_recent_errors(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent errors."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_errors = []
        
        for category_errors in self.error_history.values():
            for error_entry in category_errors:
                if error_entry['handled_at'] >= cutoff:
                    recent_errors.append(error_entry)
        
        return sorted(
            recent_errors,
            key=lambda x: x['handled_at'],
            reverse=True
        )

    def _calculate_error_trends(self) -> Dict[str, Any]:
        """Calculate error trends."""
        # Calculate error rates for different time windows
        windows = [15, 60, 240]  # minutes
        trends = {}
        
        for window in windows:
            cutoff = datetime.now() - timedelta(minutes=window)
            window_errors = sum(
                1 for errors in self.error_history.values()
                for error in errors
                if error['handled_at'] >= cutoff
            )
            
            trends[f'{window}min_rate'] = window_errors / window
        
        return trends

    def _handle_validation_error(self, error: Exception, context: ErrorContext):
        """Handle validation error."""
        logging.warning(
            f"Validation error in {context.component}: {str(error)}",
            extra={'error_context': context.__dict__}
        )

    def _handle_database_error(self, error: Exception, context: ErrorContext):
        """Handle database error."""
        logging.error(
            f"Database error in {context.component}: {str(error)}",
            extra={'error_context': context.__dict__}
        )

    def _handle_processing_error(self, error: Exception, context: ErrorContext):
        """Handle processing error."""
        logging.error(
            f"Processing error in {context.component}: {str(error)}",
            extra={'error_context': context.__dict__}
        )

    def _handle_network_error(self, error: Exception, context: ErrorContext):
        """Handle network error."""
        logging.error(
            f"Network error in {context.component}: {str(error)}",
            extra={'error_context': context.__dict__}
        )

    def _handle_system_error(self, error: Exception, context: ErrorContext):
        """Handle system error."""
        logging.critical(
            f"System error in {context.component}: {str(error)}",
            extra={'error_context': context.__dict__}
        )

    def _handle_unknown_error(self, error: Exception, context: ErrorContext):
        """Handle unknown error."""
        logging.error(
            f"Unknown error in {context.component}: {str(error)}",
            extra={'error_context': context.__dict__}
        )

    def _database_recovery_strategy(self, context: ErrorContext):
        """Recovery strategy for database errors."""
        try:
            # Implement database recovery logic
            # Example: connection retry, failover, etc.
            pass
        except Exception as e:
            logging.error(
                f"Database recovery failed: {str(e)}",
                extra={'error_context': context.__dict__}
            )

    def _network_recovery_strategy(self, context: ErrorContext):
        """Recovery strategy for network errors."""
        try:
            # Implement network recovery logic
            # Example: reconnection, alternate route, etc.
            pass
        except Exception as e:
            logging.error(
                f"Network recovery failed: {str(e)}",
                extra={'error_context': context.__dict__}
            )

# Custom Exceptions
class CircuitBreakerError(Exception):
    """Raised when circuit breaker prevents execution."""
    pass

class MaxRetriesExceededError(Exception):
    """Raised when max retries are exceeded."""
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception

# Example usage
error_manager = ErrorManager()

# Example decorator composition
def with_error_handling(
    circuit_name: str,
    error_category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Combine circuit breaker, retry, and error tracking."""
    circuit_breaker = CircuitBreaker(
        circuit_name,
        CircuitBreakerConfig()
    )
    retry_handler = RetryHandler(RetryConfig())
    
    def decorator(func):
        @wraps(func)
        @circuit_breaker
        @retry_handler
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    timestamp=datetime.now(),
                    severity=severity,
                    category=error_category,
                    component=func.__module__,
                    operation=func.__name__,
                    details={'args': args, 'kwargs': kwargs},
                    stack_trace=traceback.format_exc()
                )
                error_manager.track_error(e, context)
                raise
        return wrapper
    return decorator