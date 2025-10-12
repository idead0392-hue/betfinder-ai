"""
Agent Error Handler - Robust error handling and recovery system for BetFinder AI agents
Handles API failures, timeouts, invalid responses, and implements retry logic
"""

import asyncio
import time
import random
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import functools
import traceback
from datetime import datetime, timedelta

from agent_logger import get_logger, LogLevel, AgentEvent

class ErrorType(Enum):
    """Types of errors that can occur"""
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_INPUT = "invalid_input"
    AGENT_UNAVAILABLE = "agent_unavailable"
    PARSING_ERROR = "parsing_error"
    UNKNOWN_ERROR = "unknown_error"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FALLBACK_AGENT = "fallback_agent"
    CACHE_FALLBACK = "cache_fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"

@dataclass
class ErrorConfig:
    """Configuration for error handling behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout_seconds: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_time: int = 300  # 5 minutes
    enable_fallback: bool = True
    enable_caching: bool = True

@dataclass
class ErrorContext:
    """Context information for error handling"""
    agent_name: str
    sport: str
    session_id: str
    operation: str
    attempt: int
    start_time: datetime
    error_type: ErrorType
    original_error: Exception
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.reset_timeout
    
    def _on_success(self):
        """Reset circuit breaker on successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failure and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class AgentErrorHandler:
    """Comprehensive error handling system for agents"""
    
    def __init__(self, config: Optional[ErrorConfig] = None):
        """Initialize error handler with configuration"""
        self.config = config or ErrorConfig()
        self.logger = get_logger()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_cache: Dict[str, List[ErrorContext]] = {}
        self.response_cache: Dict[str, Any] = {}
        self.fallback_agents: Dict[str, List[str]] = {}
        
        # Error type mappings to recovery strategies
        self.error_strategies = {
            ErrorType.API_ERROR: RecoveryStrategy.EXPONENTIAL_BACKOFF,
            ErrorType.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorType.RATE_LIMIT: RecoveryStrategy.EXPONENTIAL_BACKOFF,
            ErrorType.INVALID_RESPONSE: RecoveryStrategy.RETRY,
            ErrorType.NETWORK_ERROR: RecoveryStrategy.EXPONENTIAL_BACKOFF,
            ErrorType.AUTHENTICATION_ERROR: RecoveryStrategy.FAIL_FAST,
            ErrorType.QUOTA_EXCEEDED: RecoveryStrategy.FALLBACK_AGENT,
            ErrorType.INVALID_INPUT: RecoveryStrategy.GRACEFUL_DEGRADATION,
            ErrorType.AGENT_UNAVAILABLE: RecoveryStrategy.FALLBACK_AGENT,
            ErrorType.PARSING_ERROR: RecoveryStrategy.RETRY,
            ErrorType.UNKNOWN_ERROR: RecoveryStrategy.EXPONENTIAL_BACKOFF
        }
        
        # Setup default fallback chains
        self._setup_fallback_chains()
    
    def _setup_fallback_chains(self):
        """Setup default fallback agent chains"""
        self.fallback_agents = {
            "basketball": ["local_basketball_agent", "generic_sports_agent"],
            "football": ["local_football_agent", "generic_sports_agent"],
            "soccer": ["local_soccer_agent", "generic_sports_agent"],
            "tennis": ["local_tennis_agent", "generic_sports_agent"],
            "baseball": ["local_baseball_agent", "generic_sports_agent"],
            "hockey": ["local_hockey_agent", "generic_sports_agent"],
            "csgo": ["local_esports_agent", "generic_sports_agent"],
            "league_of_legends": ["local_esports_agent", "generic_sports_agent"],
            "dota2": ["local_esports_agent", "generic_sports_agent"],
            "valorant": ["local_esports_agent", "generic_sports_agent"],
            "overwatch": ["local_esports_agent", "generic_sports_agent"],
        }
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Classify error type based on exception"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # API-specific errors
        if "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "timeout" in error_str or "timeouterror" in error_type_name:
            return ErrorType.TIMEOUT
        elif "authentication" in error_str or "401" in error_str or "unauthorized" in error_str:
            return ErrorType.AUTHENTICATION_ERROR
        elif "quota" in error_str or "limit exceeded" in error_str:
            return ErrorType.QUOTA_EXCEEDED
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "invalid response" in error_str or "malformed" in error_str:
            return ErrorType.INVALID_RESPONSE
        elif "parsing" in error_str or "json" in error_str:
            return ErrorType.PARSING_ERROR
        elif "validation" in error_str or "invalid input" in error_str:
            return ErrorType.INVALID_INPUT
        elif "unavailable" in error_str or "503" in error_str:
            return ErrorType.AGENT_UNAVAILABLE
        elif any(api_indicator in error_str for api_indicator in ["api", "openai", "request"]):
            return ErrorType.API_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def get_circuit_breaker(self, agent_name: str, sport: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent"""
        key = f"{agent_name}:{sport}"
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                reset_timeout=self.config.circuit_breaker_reset_time
            )
        return self.circuit_breakers[key]
    
    def calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate delay for retry based on attempt and error type"""
        if error_type == ErrorType.RATE_LIMIT:
            # Longer delays for rate limits
            base_delay = self.config.base_delay * 5
        else:
            base_delay = self.config.base_delay
        
        # Exponential backoff
        delay = min(
            base_delay * (self.config.backoff_multiplier ** attempt),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, error_context: ErrorContext) -> bool:
        """Determine if operation should be retried"""
        # Check max retries
        if error_context.attempt >= self.config.max_retries:
            return False
        
        # Check recovery strategy
        strategy = self.error_strategies.get(error_context.error_type, RecoveryStrategy.FAIL_FAST)
        
        if strategy == RecoveryStrategy.FAIL_FAST:
            return False
        elif strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.EXPONENTIAL_BACKOFF]:
            return True
        else:
            # For other strategies, don't retry but handle differently
            return False
    
    def get_cache_key(self, agent_name: str, sport: str, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        import hashlib
        import json
        
        cache_data = {
            'agent': agent_name,
            'sport': sport,
            'request': request_data
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response if available and not expired"""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            
            # Check expiration (cache for 1 hour)
            if 'timestamp' in cached_data:
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=1):
                    return cached_data.get('response')
        
        return None
    
    def cache_response(self, cache_key: str, response: Any):
        """Cache successful response"""
        if not self.config.enable_caching:
            return
        
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cleanup old cache entries (keep last 1000)
        if len(self.response_cache) > 1000:
            # Remove oldest 200 entries
            sorted_keys = sorted(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k]['timestamp']
            )
            for key in sorted_keys[:200]:
                del self.response_cache[key]
    
    def get_fallback_agent(self, agent_name: str, sport: str) -> Optional[str]:
        """Get next fallback agent for sport"""
        fallback_chain = self.fallback_agents.get(sport, [])
        
        # Remove current agent from chain if present
        if agent_name in fallback_chain:
            fallback_chain = [agent for agent in fallback_chain if agent != agent_name]
        
        return fallback_chain[0] if fallback_chain else None
    
    def record_error(self, error_context: ErrorContext):
        """Record error for analytics and pattern detection"""
        agent_key = f"{error_context.agent_name}:{error_context.sport}"
        
        if agent_key not in self.error_cache:
            self.error_cache[agent_key] = []
        
        self.error_cache[agent_key].append(error_context)
        
        # Keep only recent errors (last 100 per agent)
        if len(self.error_cache[agent_key]) > 100:
            self.error_cache[agent_key] = self.error_cache[agent_key][-50:]
        
        # Log error
        self.logger.log_error(
            agent_name=error_context.agent_name,
            sport=error_context.sport,
            session_id=error_context.session_id,
            error=error_context.original_error,
            context=f"Attempt {error_context.attempt} of {error_context.operation}",
            metadata={
                'error_type': error_context.error_type.value,
                'attempt': error_context.attempt,
                'operation': error_context.operation
            }
        )
    
    def create_graceful_degradation_response(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Create a graceful degradation response when agents fail"""
        return {
            'picks': [],
            'confidence': 0.0,
            'source': 'degraded',
            'error_message': f"Service temporarily unavailable. Error: {error_context.error_type.value}",
            'fallback_reason': f"Agent {error_context.agent_name} failed after {error_context.attempt} attempts",
            'timestamp': datetime.now().isoformat(),
            'degraded': True
        }

def with_error_handling(config: Optional[ErrorConfig] = None):
    """Decorator to add comprehensive error handling to agent functions"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context from function arguments
            agent_name = kwargs.get('agent_name', 'unknown')
            sport = kwargs.get('sport', 'unknown') 
            session_id = kwargs.get('session_id', 'unknown')
            
            error_handler = AgentErrorHandler(config)
            logger = get_logger()
            
            # Get cache key if request data is available
            cache_key = None
            if 'request_data' in kwargs:
                cache_key = error_handler.get_cache_key(agent_name, sport, kwargs['request_data'])
                
                # Try cache first
                cached_response = error_handler.get_cached_response(cache_key)
                if cached_response:
                    logger.log_event(
                        event_type=AgentEvent.RESPONSE_PROCESSED,
                        level=LogLevel.INFO,
                        agent_name=agent_name,
                        sport=sport,
                        session_id=session_id,
                        metadata={'source': 'cache', 'cache_key': cache_key}
                    )
                    return cached_response
            
            # Get circuit breaker
            circuit_breaker = error_handler.get_circuit_breaker(agent_name, sport)
            
            attempt = 0
            start_time = datetime.now()
            
            while attempt < error_handler.config.max_retries + 1:
                try:
                    attempt += 1
                    
                    # Log attempt start
                    if attempt > 1:
                        logger.log_event(
                            event_type=AgentEvent.REQUEST_START,
                            level=LogLevel.INFO,
                            agent_name=agent_name,
                            sport=sport,
                            session_id=session_id,
                            metadata={'attempt': attempt, 'retry': True}
                        )
                    
                    # Execute function with circuit breaker
                    result = circuit_breaker.call(func, *args, **kwargs)
                    
                    # Cache successful response
                    if cache_key and result:
                        error_handler.cache_response(cache_key, result)
                    
                    # Log success
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    logger.log_request_success(
                        agent_name=agent_name,
                        sport=sport,
                        session_id=session_id,
                        response_data={'success': True, 'attempt': attempt},
                        response_time_ms=response_time,
                        metadata={'total_attempts': attempt}
                    )
                    
                    return result
                
                except Exception as e:
                    # Classify error
                    error_type = error_handler.classify_error(e)
                    
                    # Create error context
                    error_context = ErrorContext(
                        agent_name=agent_name,
                        sport=sport,
                        session_id=session_id,
                        operation=func.__name__,
                        attempt=attempt,
                        start_time=start_time,
                        error_type=error_type,
                        original_error=e,
                        request_data=kwargs.get('request_data'),
                        response_data=None
                    )
                    
                    # Record error
                    error_handler.record_error(error_context)
                    
                    # Check if should retry
                    if error_handler.should_retry(error_context):
                        delay = error_handler.calculate_delay(attempt - 1, error_type)
                        
                        logger.log_event(
                            event_type=AgentEvent.ERROR_HANDLED,
                            level=LogLevel.WARNING,
                            agent_name=agent_name,
                            sport=sport,
                            session_id=session_id,
                            error_type=type(e).__name__,
                            error_message=f"Retrying in {delay:.2f}s (attempt {attempt}/{error_handler.config.max_retries})",
                            metadata={
                                'error_type': error_type.value,
                                'retry_delay': delay,
                                'attempt': attempt
                            }
                        )
                        
                        time.sleep(delay)
                        continue
                    else:
                        # No more retries, check for fallback strategies
                        strategy = error_handler.error_strategies.get(error_type, RecoveryStrategy.FAIL_FAST)
                        
                        if strategy == RecoveryStrategy.FALLBACK_AGENT and error_handler.config.enable_fallback:
                            fallback_agent = error_handler.get_fallback_agent(agent_name, sport)
                            if fallback_agent:
                                logger.log_fallback(
                                    agent_name=agent_name,
                                    sport=sport,
                                    session_id=session_id,
                                    reason=f"Primary agent failed: {error_type.value}",
                                    fallback_agent=fallback_agent,
                                    metadata={'original_error': str(e)}
                                )
                                
                                # Try fallback agent (this would need integration with agent manager)
                                # For now, return graceful degradation
                                return error_handler.create_graceful_degradation_response(error_context)
                        
                        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                            return error_handler.create_graceful_degradation_response(error_context)
                        
                        elif strategy == RecoveryStrategy.CACHE_FALLBACK:
                            # Try to return any cached response, even if expired
                            if cache_key and cache_key in error_handler.response_cache:
                                cached_response = error_handler.response_cache[cache_key]['response']
                                logger.log_event(
                                    event_type=AgentEvent.RESPONSE_PROCESSED,
                                    level=LogLevel.WARNING,
                                    agent_name=agent_name,
                                    sport=sport,
                                    session_id=session_id,
                                    metadata={'source': 'expired_cache', 'reason': 'fallback'}
                                )
                                return cached_response
                        
                        # Final failure
                        logger.log_request_failure(
                            agent_name=agent_name,
                            sport=sport,
                            session_id=session_id,
                            error=e,
                            metadata={
                                'error_type': error_type.value,
                                'total_attempts': attempt,
                                'strategy': strategy.value
                            }
                        )
                        
                        # Re-raise the original exception
                        raise e
            
            # Should not reach here
            raise Exception(f"Max retries exceeded for {func.__name__}")
        
        return wrapper
    return decorator

def with_timeout(timeout_seconds: float = 30.0):
    """Decorator to add timeout protection to agent functions"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set timeout alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except TimeoutError:
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        
        return wrapper
    return decorator

# Async version of error handling decorator
def with_async_error_handling(config: Optional[ErrorConfig] = None):
    """Async decorator to add comprehensive error handling to agent functions"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Similar implementation to sync version but with async/await
            agent_name = kwargs.get('agent_name', 'unknown')
            sport = kwargs.get('sport', 'unknown') 
            session_id = kwargs.get('session_id', 'unknown')
            
            error_handler = AgentErrorHandler(config)
            logger = get_logger()
            
            attempt = 0
            start_time = datetime.now()
            
            while attempt < error_handler.config.max_retries + 1:
                try:
                    attempt += 1
                    result = await func(*args, **kwargs)
                    
                    # Log success
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    logger.log_request_success(
                        agent_name=agent_name,
                        sport=sport,
                        session_id=session_id,
                        response_data={'success': True, 'attempt': attempt},
                        response_time_ms=response_time
                    )
                    
                    return result
                
                except Exception as e:
                    error_type = error_handler.classify_error(e)
                    
                    error_context = ErrorContext(
                        agent_name=agent_name,
                        sport=sport,
                        session_id=session_id,
                        operation=func.__name__,
                        attempt=attempt,
                        start_time=start_time,
                        error_type=error_type,
                        original_error=e
                    )
                    
                    error_handler.record_error(error_context)
                    
                    if error_handler.should_retry(error_context):
                        delay = error_handler.calculate_delay(attempt - 1, error_type)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.log_request_failure(
                            agent_name=agent_name,
                            sport=sport,
                            session_id=session_id,
                            error=e
                        )
                        raise e
            
            raise Exception(f"Max retries exceeded for {func.__name__}")
        
        return wrapper
    return decorator