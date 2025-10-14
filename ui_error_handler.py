"""
UI Error Handler - Streamlit components for graceful error handling and user feedback
Provides retry mechanisms, fallback displays, and user-friendly error messages
"""

import streamlit as st
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from agent_logger import get_logger, LogLevel, AgentEvent
from agent_error_handler import AgentErrorHandler
from agent_monitor import get_monitor

class UIErrorLevel(Enum):
    """UI error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class UIErrorType(Enum):
    """Types of UI errors"""
    AGENT_FAILURE = "agent_failure"
    DATA_LOADING = "data_loading"
    NETWORK_ISSUE = "configuration"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    SYSTEM_ERROR = "system_error"

@dataclass
class UIErrorMessage:
    """Structured error message for UI display"""
    title: str
    message: str
    error_type: UIErrorType
    level: UIErrorLevel
    retry_available: bool = True
    fallback_available: bool = False
    technical_details: Optional[str] = None
    resolution_steps: Optional[List[str]] = None
    estimated_fix_time: Optional[str] = None

class UIErrorHandler:
    """Streamlit UI error handling system"""
    
    def __init__(self):
        self.logger = get_logger()
        self.monitor = get_monitor()
        self.error_handler = AgentErrorHandler()
        
        # Error message templates
        self.error_templates = self._initialize_error_templates()
        
        # Session state for error tracking
        if 'ui_error_history' not in st.session_state:
            st.session_state.ui_error_history = []
        if 'ui_retry_counts' not in st.session_state:
            st.session_state.ui_retry_counts = {}
        if 'ui_fallback_mode' not in st.session_state:
            st.session_state.ui_fallback_mode = False
    
    def _initialize_error_templates(self) -> Dict[UIErrorType, UIErrorMessage]:
        """Initialize error message templates"""
        return {
            UIErrorType.AGENT_FAILURE: UIErrorMessage(
                title="🤖 Agent Service Issue",
                message="The AI agent is temporarily unavailable. This usually resolves quickly.",
                error_type=UIErrorType.AGENT_FAILURE,
                level=UIErrorLevel.WARNING,
                retry_available=True,
                fallback_available=True,
                resolution_steps=[
                    "Wait a moment and try again",
                    "Check if your API key is valid",
                    "Try using fallback mode"
                ],
                estimated_fix_time="1-2 minutes"
            ),
            
            UIErrorType.DATA_LOADING: UIErrorMessage(
                title="📊 Data Loading Issue",
                message="Unable to load the latest sports data. Some information may be outdated.",
                error_type=UIErrorType.DATA_LOADING,
                level=UIErrorLevel.WARNING,
                retry_available=True,
                fallback_available=True,
                resolution_steps=[
                    "Refresh the page",
                    "Check your internet connection",
                    "Try again in a few minutes"
                ],
                estimated_fix_time="30 seconds"
            ),
            
            UIErrorType.NETWORK_ISSUE: UIErrorMessage(
                title="🌐 Network Connection Problem",
                message="Cannot connect to external services. Please check your internet connection.",
                error_type=UIErrorType.NETWORK_ISSUE,
                level=UIErrorLevel.ERROR,
                retry_available=True,
                fallback_available=True,
                resolution_steps=[
                    "Check your internet connection",
                    "Try refreshing the page",
                    "Wait a moment and retry"
                ],
                estimated_fix_time="Variable"
            ),
            
            UIErrorType.RATE_LIMIT: UIErrorMessage(
                title="⏱️ Rate Limit Reached",
                message="Too many requests sent. Please wait before trying again.",
                error_type=UIErrorType.RATE_LIMIT,
                level=UIErrorLevel.WARNING,
                retry_available=True,
                fallback_available=True,
                resolution_steps=[
                    "Wait for the cooldown period",
                    "Reduce request frequency",
                    "Try again in a few minutes"
                ],
                estimated_fix_time="2-5 minutes"
            ),
            
            UIErrorType.TIMEOUT: UIErrorMessage(
                title="⏰ Request Timeout",
                message="The request took too long to complete. The service may be busy.",
                error_type=UIErrorType.TIMEOUT,
                level=UIErrorLevel.WARNING,
                retry_available=True,
                fallback_available=True,
                resolution_steps=[
                    "Try again immediately",
                    "Reduce the complexity of your request",
                    "Check system status"
                ],
                estimated_fix_time="Immediate"
            ),
            
            UIErrorType.CONFIGURATION: UIErrorMessage(
                title="⚙️ Configuration Issue",
                message="There's a problem with the system configuration.",
                error_type=UIErrorType.CONFIGURATION,
                level=UIErrorLevel.ERROR,
                retry_available=False,
                fallback_available=True,
                resolution_steps=[
                    "Check your API keys in the sidebar",
                    "Verify environment variables",
                    "Contact support if the issue persists"
                ],
                estimated_fix_time="User action required"
            ),
            
            UIErrorType.INVALID_INPUT: UIErrorMessage(
                title="📝 Invalid Input",
                message="The provided input is not valid or is missing required information.",
                error_type=UIErrorType.INVALID_INPUT,
                level=UIErrorLevel.INFO,
                retry_available=False,
                fallback_available=False,
                resolution_steps=[
                    "Check your input for errors",
                    "Ensure all required fields are filled",
                    "Review the input format requirements"
                ],
                estimated_fix_time="Immediate"
            ),
            
            UIErrorType.SYSTEM_ERROR: UIErrorMessage(
                title="🚨 System Error",
                message="An unexpected system error occurred. Our team has been notified.",
                error_type=UIErrorType.SYSTEM_ERROR,
                level=UIErrorLevel.CRITICAL,
                retry_available=True,
                fallback_available=True,
                resolution_steps=[
                    "Try refreshing the page",
                    "Clear your browser cache",
                    "Contact support with error details"
                ],
                estimated_fix_time="5-10 minutes"
            )
        }
    
    def classify_error(self, error: Exception, context: str = "") -> UIErrorType:
        """Classify error for appropriate UI handling"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Map common errors to UI error types
        if "rate limit" in error_str or "429" in error_str:
            return UIErrorType.RATE_LIMIT
        elif "timeout" in error_str or "timeouterror" in error_type_name:
            return UIErrorType.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return UIErrorType.NETWORK_ISSUE
        elif "agent" in context.lower() or "openai" in error_str:
            return UIErrorType.AGENT_FAILURE
        elif "data" in context.lower() or "loading" in context.lower():
            return UIErrorType.DATA_LOADING
        elif "config" in error_str or "api key" in error_str or "unauthorized" in error_str:
            return UIErrorType.CONFIGURATION
        elif "validation" in error_str or "invalid" in error_str:
            return UIErrorType.INVALID_INPUT
        else:
            return UIErrorType.SYSTEM_ERROR
    
    def display_error(self, error: Exception, context: str = "",
                     show_technical_details: bool = False,
                     session_id: str = None) -> bool:
        """Display error message with retry options. Returns True if user wants to retry."""
        
        # Classify error
        ui_error_type = self.classify_error(error, context)
        
        # Get error template
        error_template = self.error_templates[ui_error_type]
        
        # Create error message
        error_msg = UIErrorMessage(
            title=error_template.title,
            message=error_template.message,
            error_type=ui_error_type,
            level=error_template.level,
            retry_available=error_template.retry_available,
            fallback_available=error_template.fallback_available,
            technical_details=str(error) if show_technical_details else None,
            resolution_steps=error_template.resolution_steps,
            estimated_fix_time=error_template.estimated_fix_time
        )
        
        # Log error
        if session_id:
            self.logger.log_event(
                event_type=AgentEvent.ERROR_HANDLED,
                level=LogLevel.ERROR,
                agent_name="ui_system",
                sport="general",
                session_id=session_id,
                error_type=type(error).__name__,
                error_message=str(error),
                metadata={
                    'ui_error_type': ui_error_type.value,
                    'context': context,
                    'level': error_template.level.value
                }
            )
        
        # Store in session history
        st.session_state.ui_error_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error_type': ui_error_type.value,
            'message': str(error),
            'context': context
        })
        
        # Display error UI
        return self._render_error_ui(error_msg, context, session_id)
    
    def _render_error_ui(self, error_msg: UIErrorMessage, context: str,
                        session_id: str = None) -> bool:
        """Render the error UI and handle user interactions"""
        
        # Choose appropriate Streamlit alert type
        alert_type = {
            UIErrorLevel.INFO: st.info,
            UIErrorLevel.WARNING: st.warning,
            UIErrorLevel.ERROR: st.error,
            UIErrorLevel.CRITICAL: st.error
        }[error_msg.level]
        
        # Display main error message
        with alert_type(error_msg.title):
            st.write(error_msg.message)
            
            # Show resolution steps
            if error_msg.resolution_steps:
                st.write("**Possible solutions:**")
                for step in error_msg.resolution_steps:
                    st.write(f"• {step}")
            
            # Show estimated fix time
            if error_msg.estimated_fix_time:
                st.write(f"**Estimated resolution time:** {error_msg.estimated_fix_time}")
        
        # Show technical details if requested
        if error_msg.technical_details:
            with st.expander("Technical Details"):
                st.code(error_msg.technical_details)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        retry_clicked = False
        fallback_clicked = False
        
        with col1:
            if error_msg.retry_available:
                retry_clicked = st.button("🔄 Retry", key=f"retry_{context}_{time.time()}")
        
        with col2:
            if error_msg.fallback_available:
                fallback_clicked = st.button("🔧 Use Fallback", key=f"fallback_{context}_{time.time()}")
        
        with col3:
            if st.button("ℹ️ Show Details", key=f"details_{context}_{time.time()}"):
                st.session_state[f"show_details_{context}"] = True
                st.rerun()
        
        # Handle retry
        if retry_clicked:
            self._handle_retry(error_msg.error_type, context, session_id)
            return True
        
        # Handle fallback
        if fallback_clicked:
            self._handle_fallback(error_msg.error_type, context, session_id)
            return False
        
        return False
    
    def _handle_retry(self, error_type: UIErrorType, context: str, session_id: str = None):
        """Handle retry action"""
        # Track retry count
        retry_key = f"{error_type.value}_{context}"
        if retry_key not in st.session_state.ui_retry_counts:
            st.session_state.ui_retry_counts[retry_key] = 0
        
        st.session_state.ui_retry_counts[retry_key] += 1
        
        # Log retry
        if session_id:
            self.logger.log_event(
                event_type=AgentEvent.ERROR_HANDLED,
                level=LogLevel.INFO,
                agent_name="ui_system",
                sport="general",
                session_id=session_id,
                metadata={
                    'action': 'retry',
                    'error_type': error_type.value,
                    'context': context,
                    'retry_count': st.session_state.ui_retry_counts[retry_key]
                }
            )
        
        # Show retry message
        if error_type == UIErrorType.RATE_LIMIT:
            st.info("⏳ Waiting for rate limit reset...")
            time.sleep(2)  # Brief pause for rate limits
        elif error_type == UIErrorType.TIMEOUT:
            st.info("🔄 Retrying with optimized settings...")
        else:
            st.info("🔄 Retrying operation...")
    
    def _handle_fallback(self, error_type: UIErrorType, context: str, session_id: str = None):
        """Handle fallback mode activation"""
        st.session_state.ui_fallback_mode = True
        
        # Log fallback activation
        if session_id:
            self.logger.log_event(
                event_type=AgentEvent.FALLBACK_TRIGGERED,
                level=LogLevel.WARNING,
                agent_name="ui_system",
                sport="general",
                session_id=session_id,
                metadata={
                    'action': 'fallback_activated',
                    'error_type': error_type.value,
                    'context': context
                }
            )
        
        # Show fallback message
        st.info("🔧 Switched to fallback mode. Some features may be limited but basic functionality is available.")
    
    def with_error_handling(self, operation_name: str, session_id: str = None):
        """Decorator for wrapping operations with error handling"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Display error and handle retry
                    retry = self.display_error(
                        error=e,
                        context=operation_name,
                        session_id=session_id
                    )
                    
                    if retry:
                        # User wants to retry
                        try:
                            return func(*args, **kwargs)
                        except Exception as retry_error:
                            # Second failure
                            self.display_error(
                                error=retry_error,
                                context=f"{operation_name}_retry",
                                session_id=session_id
                            )
                            return None
                    return None
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, operation_name: str,
                    fallback_func: Callable = None, session_id: str = None,
                    max_retries: int = 2) -> Any:
        """Safely execute a function with error handling and retries"""
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            
            except Exception as e:
                is_last_attempt = attempt == max_retries
                
                if is_last_attempt:
                    # Final attempt failed
                    retry = self.display_error(
                        error=e,
                        context=f"{operation_name}_final",
                        session_id=session_id
                    )
                    
                    # Try fallback if available and user doesn't want to retry
                    if not retry and fallback_func:
                        st.warning("🔧 Attempting fallback operation...")
                        try:
                            return fallback_func()
                        except Exception as fallback_error:
                            self.display_error(
                                error=fallback_error,
                                context=f"{operation_name}_fallback",
                                session_id=session_id
                            )
                    
                    return None
                else:
                    # Intermediate attempt failed, show brief error and retry
                    st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)  # Brief pause before retry
        
        return None
    
    def display_loading_with_fallback(self, operation_name: str, 
                                    timeout_seconds: int = 30,
                                    fallback_message: str = None) -> bool:
        """Display loading spinner with timeout and fallback options"""
        
        fallback_msg = fallback_message or f"Loading {operation_name} is taking longer than expected"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        for i in range(timeout_seconds):
            elapsed = time.time() - start_time
            progress = min(elapsed / timeout_seconds, 1.0)
            
            progress_bar.progress(progress)
            status_text.text(f"Loading {operation_name}... ({i+1}s)")
            
            time.sleep(1)
            
            # Check if operation completed (this would need to be integrated with actual operations)
            # For now, just simulate
            if i > timeout_seconds * 0.8:  # 80% through timeout
                break
        
        # Timeout reached
        progress_bar.empty()
        status_text.empty()
        
        st.warning(f"⏰ {fallback_msg}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wait_more = st.button("⏳ Wait More", key=f"wait_{operation_name}_{time.time()}")
        
        with col2:
            use_fallback = st.button("🔧 Use Fallback", key=f"fallback_{operation_name}_{time.time()}")
        
        if wait_more:
            return self.display_loading_with_fallback(operation_name, timeout_seconds, fallback_message)
        
        return not use_fallback  # Return True if should continue, False if should use fallback
    
    def display_system_status(self):
        """Display current system status and health"""
        
        # Get system health from monitor
        try:
            health = self.monitor.get_system_health()
            
            # Display status
            status_color = {
                'healthy': '🟢',
                'warning': '🟡', 
                'critical': '🔴'
            }.get(health['status'], '🔵')
            
            st.sidebar.markdown(f"### {status_color} System Status")
            
            # Status details
            with st.sidebar.expander("System Details"):
                st.write(f"**Status:** {health['status'].title()}")
                st.write(f"**Active Agents:** {health['healthy_agents']}/{health['total_agents']}")
                st.write(f"**Uptime:** {health['uptime_percentage']:.1f}%")
                st.write(f"**Avg Response:** {health['avg_response_time']:.0f}ms")
                
                if health['error_agents'] > 0:
                    st.write(f"**Agents with Issues:** {health['error_agents']}")
                
                st.write(f"**Last Updated:** {health.get('last_updated', 'Unknown')}")
            
            # Show fallback mode status
            if st.session_state.ui_fallback_mode:
                st.sidebar.warning("🔧 Fallback Mode Active")
                if st.sidebar.button("Exit Fallback Mode"):
                    st.session_state.ui_fallback_mode = False
                    st.rerun()
        
        except Exception as e:
            st.sidebar.error("❌ Status Check Failed")
            st.sidebar.write(f"Error: {str(e)}")
    
    def display_error_history(self):
        """Display recent error history for debugging"""
        
        if not st.session_state.ui_error_history:
            return
        
        with st.sidebar.expander("Recent Errors"):
            recent_errors = st.session_state.ui_error_history[-5:]  # Last 5 errors
            
            for i, error in enumerate(reversed(recent_errors)):
                timestamp = error['timestamp']
                error_type = error['error_type']
                context = error['context']
                
                st.write(f"**{i+1}.** {error_type}")
                st.write(f"Context: {context}")
                st.write(f"Time: {timestamp[:19]}")  # Remove timezone for brevity
                st.write("---")
            
            if st.button("Clear Error History"):
                st.session_state.ui_error_history = []
                st.rerun()
    
    def create_error_boundary(self, component_name: str, session_id: str = None):
        """Create an error boundary context manager for Streamlit components"""
        
        class ErrorBoundary:
            def __init__(self, handler, name, session_id):
                self.handler = handler
                self.name = name
                self.session_id = session_id
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    # Error occurred
                    self.handler.display_error(
                        error=exc_val,
                        context=self.name,
                        session_id=self.session_id
                    )
                    return True  # Suppress the exception
                return False
        
        return ErrorBoundary(self, component_name, session_id)
    
    def display_retry_prompt(self, operation_name: str, 
                           current_attempt: int, max_attempts: int,
                           error: Exception) -> bool:
        """Display retry prompt for failed operations"""
        
        st.warning(f"⚠️ {operation_name} failed (attempt {current_attempt}/{max_attempts})")
        st.write(f"Error: {str(error)}")
        
        if current_attempt < max_attempts:
            col1, col2 = st.columns(2)
            
            with col1:
                retry = st.button(f"🔄 Retry ({max_attempts - current_attempt} left)")
            
            with col2:
                cancel = st.button("❌ Cancel")
            
            if retry:
                return True
            elif cancel:
                return False
        else:
            st.error(f"❌ {operation_name} failed after {max_attempts} attempts")
            return False
        
        return False
    
    def show_maintenance_mode(self, message: str = None):
        """Display maintenance mode message"""
        default_message = "🔧 The system is currently undergoing maintenance. Please try again later."
        
        st.error(message or default_message)
        
        st.info("""
        **What you can do:**
        • Check back in a few minutes
        • Try using cached data if available
        • Contact support if this persists
        """)
    
    def create_status_indicator(self, status: str, details: str = "") -> str:
        """Create a status indicator for UI elements"""
        
        indicators = {
            'healthy': '🟢',
            'good': '🟢',
            'warning': '🟡',
            'error': '🔴',
            'critical': '🔴',
            'unknown': '🔵',
            'loading': '🔄',
            'offline': '⚪'
        }
        
        icon = indicators.get(status.lower(), '🔵')
        
        if details:
            return f"{icon} {status.title()} - {details}"
        else:
            return f"{icon} {status.title()}"

# Global UI error handler instance
_global_ui_error_handler: Optional[UIErrorHandler] = None

def get_ui_error_handler() -> UIErrorHandler:
    """Get the global UI error handler instance"""
    global _global_ui_error_handler
    if _global_ui_error_handler is None:
        _global_ui_error_handler = UIErrorHandler()
    return _global_ui_error_handler

# Convenience decorators and functions for common use cases
def with_ui_error_handling(operation_name: str, session_id: str = None):
    """Decorator for adding UI error handling to functions"""
    handler = get_ui_error_handler()
    return handler.with_error_handling(operation_name, session_id)

def safe_streamlit_operation(func: Callable, operation_name: str,
                           fallback_func: Callable = None,
                           session_id: str = None) -> Any:
    """Execute a Streamlit operation with error handling"""
    handler = get_ui_error_handler()
    return handler.safe_execute(func, operation_name, fallback_func, session_id)

def display_error_message(error: Exception, context: str = "",
                         session_id: str = None) -> bool:
    """Display an error message in the UI"""
    handler = get_ui_error_handler()
    return handler.display_error(error, context, session_id=session_id)