"""
Agent Logger - Comprehensive logging system for BetFinder AI agents
Tracks requests, responses, errors, performance metrics, and outcomes
"""

import logging
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import pandas as pd

class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AgentEvent(Enum):
    """Types of agent events to log"""
    REQUEST_START = "request_start"
    REQUEST_SUCCESS = "request_success"
    REQUEST_FAILURE = "request_failure"
    RESPONSE_PROCESSED = "response_processed"
    PICK_GENERATED = "pick_generated"
    PICK_OUTCOME = "pick_outcome"
    ERROR_HANDLED = "error_handled"
    FALLBACK_TRIGGERED = "fallback_triggered"
    PERFORMANCE_METRIC = "performance_metric"

@dataclass
class AgentLogEntry:
    """Structured log entry for agent operations"""
    timestamp: str
    event_id: str
    event_type: AgentEvent
    level: LogLevel
    agent_name: str
    sport: str
    session_id: str
    user_id: Optional[str] = None
    
    # Request details
    request_data: Optional[Dict[str, Any]] = None
    request_size: Optional[int] = None
    
    # Response details
    response_data: Optional[Dict[str, Any]] = None
    response_size: Optional[int] = None
    response_time_ms: Optional[float] = None
    
    # Pick details
    pick_id: Optional[str] = None
    pick_confidence: Optional[float] = None
    pick_type: Optional[str] = None
    pick_value: Optional[float] = None
    expected_value: Optional[float] = None
    
    # Outcome tracking
    outcome: Optional[str] = None  # 'win', 'loss', 'push', 'pending'
    actual_value: Optional[float] = None
    profit_loss: Optional[float] = None
    
    # Error details
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Performance metrics
    api_version: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    
    # Additional context
    metadata: Optional[Dict[str, Any]] = None

class AgentLogger:
    """Comprehensive logging system for agent operations"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: LogLevel = LogLevel.INFO,
                 max_log_files: int = 30,
                 max_file_size_mb: int = 100):
        """Initialize the agent logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = log_level
        self.max_log_files = max_log_files
        self.max_file_size_mb = max_file_size_mb
        
        # Setup file logging
        self._setup_file_logging()
        
        # In-memory storage for real-time analytics
        self.recent_logs: List[AgentLogEntry] = []
        self.session_cache: Dict[str, List[AgentLogEntry]] = {}
        self.performance_cache: Dict[str, Dict[str, Any]] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize performance tracking
        self._init_performance_tracking()
    
    def _setup_file_logging(self):
        """Setup file-based logging with rotation"""
        # Create date-based log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"agent_logs_{today}.jsonl"
        
        # Setup Python logger
        self.logger = logging.getLogger("agent_logger")
        self.logger.setLevel(getattr(logging, self.log_level.value))
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(getattr(logging, self.log_level.value))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Cleanup old logs
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove old log files to manage disk space"""
        log_files = sorted(self.log_dir.glob("agent_logs_*.jsonl"))
        if len(log_files) > self.max_log_files:
            for old_file in log_files[:-self.max_log_files]:
                old_file.unlink()
    
    def _init_performance_tracking(self):
        """Initialize performance tracking metrics"""
        self.performance_cache = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'avg_response_time': 0.0,
            'total_cost': 0.0,
            'picks_generated': 0,
            'picks_won': 0,
            'picks_lost': 0,
            'total_profit': 0.0,
            'agent_performance': {},
            'error_counts': {},
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def create_session(self, agent_name: str, sport: str, user_id: Optional[str] = None) -> str:
        """Create a new logging session"""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'agent_name': agent_name,
            'sport': sport,
            'user_id': user_id,
            'start_time': datetime.now(timezone.utc),
            'requests': 0,
            'picks_generated': 0,
            'errors': 0
        }
        self.session_cache[session_id] = []
        return session_id
    
    def log_event(self, 
                  event_type: AgentEvent,
                  level: LogLevel,
                  agent_name: str,
                  sport: str,
                  session_id: str,
                  **kwargs) -> str:
        """Log an agent event"""
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        # Create log entry
        log_entry = AgentLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id=event_id,
            event_type=event_type,
            level=level,
            agent_name=agent_name,
            sport=sport,
            session_id=session_id,
            **kwargs
        )
        
        # Store in memory
        self.recent_logs.append(log_entry)
        if session_id in self.session_cache:
            self.session_cache[session_id].append(log_entry)
        
        # Keep recent logs manageable
        if len(self.recent_logs) > 1000:
            self.recent_logs = self.recent_logs[-500:]
        
        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(log_entry), default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write log entry: {e}")
        
        # Update performance metrics
        self._update_performance_metrics(log_entry)
        
        # Log to Python logger
        log_message = f"[{agent_name}:{sport}] {event_type.value}"
        if kwargs.get('error_message'):
            log_message += f" - {kwargs['error_message']}"
        
        getattr(self.logger, level.value.lower())(log_message)
        
        return event_id
    
    def log_request_start(self, agent_name: str, sport: str, session_id: str, 
                         request_data: Dict[str, Any], **kwargs) -> str:
        """Log the start of an agent request"""
        return self.log_event(
            event_type=AgentEvent.REQUEST_START,
            level=LogLevel.INFO,
            agent_name=agent_name,
            sport=sport,
            session_id=session_id,
            request_data=request_data,
            request_size=len(json.dumps(request_data)),
            **kwargs
        )
    
    def log_request_success(self, agent_name: str, sport: str, session_id: str,
                           response_data: Dict[str, Any], response_time_ms: float,
                           **kwargs) -> str:
        """Log a successful agent request"""
        return self.log_event(
            event_type=AgentEvent.REQUEST_SUCCESS,
            level=LogLevel.INFO,
            agent_name=agent_name,
            sport=sport,
            session_id=session_id,
            response_data=response_data,
            response_size=len(json.dumps(response_data)),
            response_time_ms=response_time_ms,
            **kwargs
        )
    
    def log_request_failure(self, agent_name: str, sport: str, session_id: str,
                           error: Exception, **kwargs) -> str:
        """Log a failed agent request"""
        return self.log_event(
            event_type=AgentEvent.REQUEST_FAILURE,
            level=LogLevel.ERROR,
            agent_name=agent_name,
            sport=sport,
            session_id=session_id,
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            **kwargs
        )
    
    def log_pick_generated(self, agent_name: str, sport: str, session_id: str,
                          pick_id: str, pick_data: Dict[str, Any], **kwargs) -> str:
        """Log when a pick is generated"""
        return self.log_event(
            event_type=AgentEvent.PICK_GENERATED,
            level=LogLevel.INFO,
            agent_name=agent_name,
            sport=sport,
            session_id=session_id,
            pick_id=pick_id,
            pick_confidence=pick_data.get('confidence'),
            pick_type=pick_data.get('type'),
            pick_value=pick_data.get('value'),
            expected_value=pick_data.get('expected_value'),
            metadata=pick_data,
            **kwargs
        )
    
    def log_pick_outcome(self, pick_id: str, outcome: str, actual_value: float,
                        profit_loss: float, **kwargs) -> str:
        """Log the outcome of a pick"""
        # Find the original pick log entry
        pick_entry = None
        for entry in reversed(self.recent_logs):
            if entry.pick_id == pick_id and entry.event_type == AgentEvent.PICK_GENERATED:
                pick_entry = entry
                break
        
        if pick_entry:
            return self.log_event(
                event_type=AgentEvent.PICK_OUTCOME,
                level=LogLevel.INFO,
                agent_name=pick_entry.agent_name,
                sport=pick_entry.sport,
                session_id=pick_entry.session_id,
                pick_id=pick_id,
                outcome=outcome,
                actual_value=actual_value,
                profit_loss=profit_loss,
                **kwargs
            )
        else:
            return self.log_event(
                event_type=AgentEvent.PICK_OUTCOME,
                level=LogLevel.WARNING,
                agent_name="unknown",
                sport="unknown",
                session_id="unknown",
                pick_id=pick_id,
                outcome=outcome,
                actual_value=actual_value,
                profit_loss=profit_loss,
                error_message="Original pick entry not found",
                **kwargs
            )
    
    def log_error(self, agent_name: str, sport: str, session_id: str,
                  error: Exception, context: str = "", **kwargs) -> str:
        """Log an error with context"""
        return self.log_event(
            event_type=AgentEvent.ERROR_HANDLED,
            level=LogLevel.ERROR,
            agent_name=agent_name,
            sport=sport,
            session_id=session_id,
            error_type=type(error).__name__,
            error_message=f"{context}: {str(error)}" if context else str(error),
            error_traceback=traceback.format_exc(),
            **kwargs
        )
    
    def log_fallback(self, agent_name: str, sport: str, session_id: str,
                    reason: str, fallback_agent: str, **kwargs) -> str:
        """Log when fallback agent is triggered"""
        return self.log_event(
            event_type=AgentEvent.FALLBACK_TRIGGERED,
            level=LogLevel.WARNING,
            agent_name=agent_name,
            sport=sport,
            session_id=session_id,
            error_message=reason,
            metadata={'fallback_agent': fallback_agent},
            **kwargs
        )
    
    def _update_performance_metrics(self, log_entry: AgentLogEntry):
        """Update performance tracking metrics"""
        try:
            # Update request counts
            if log_entry.event_type == AgentEvent.REQUEST_START:
                self.performance_cache['requests_total'] += 1
                
                # Update session metrics
                if log_entry.session_id in self.active_sessions:
                    self.active_sessions[log_entry.session_id]['requests'] += 1
            
            elif log_entry.event_type == AgentEvent.REQUEST_SUCCESS:
                self.performance_cache['requests_success'] += 1
                
                # Update response time
                if log_entry.response_time_ms:
                    current_avg = self.performance_cache['avg_response_time']
                    total_success = self.performance_cache['requests_success']
                    new_avg = ((current_avg * (total_success - 1)) + log_entry.response_time_ms) / total_success
                    self.performance_cache['avg_response_time'] = new_avg
            
            elif log_entry.event_type == AgentEvent.REQUEST_FAILURE:
                self.performance_cache['requests_failed'] += 1
                
                # Update error counts
                error_type = log_entry.error_type or 'Unknown'
                if error_type not in self.performance_cache['error_counts']:
                    self.performance_cache['error_counts'][error_type] = 0
                self.performance_cache['error_counts'][error_type] += 1
                
                # Update session metrics
                if log_entry.session_id in self.active_sessions:
                    self.active_sessions[log_entry.session_id]['errors'] += 1
            
            elif log_entry.event_type == AgentEvent.PICK_GENERATED:
                self.performance_cache['picks_generated'] += 1
                
                # Update session metrics
                if log_entry.session_id in self.active_sessions:
                    self.active_sessions[log_entry.session_id]['picks_generated'] += 1
                
                # Update agent-specific metrics
                agent_key = f"{log_entry.agent_name}:{log_entry.sport}"
                if agent_key not in self.performance_cache['agent_performance']:
                    self.performance_cache['agent_performance'][agent_key] = {
                        'picks_generated': 0,
                        'picks_won': 0,
                        'picks_lost': 0,
                        'total_profit': 0.0,
                        'avg_confidence': 0.0,
                        'last_active': None
                    }
                
                agent_perf = self.performance_cache['agent_performance'][agent_key]
                agent_perf['picks_generated'] += 1
                agent_perf['last_active'] = log_entry.timestamp
                
                # Update confidence average
                if log_entry.pick_confidence:
                    current_avg = agent_perf['avg_confidence']
                    total_picks = agent_perf['picks_generated']
                    new_avg = ((current_avg * (total_picks - 1)) + log_entry.pick_confidence) / total_picks
                    agent_perf['avg_confidence'] = new_avg
            
            elif log_entry.event_type == AgentEvent.PICK_OUTCOME:
                if log_entry.outcome == 'win':
                    self.performance_cache['picks_won'] += 1
                elif log_entry.outcome == 'loss':
                    self.performance_cache['picks_lost'] += 1
                
                if log_entry.profit_loss:
                    self.performance_cache['total_profit'] += log_entry.profit_loss
                
                # Update agent-specific outcomes
                # Find the agent from the original pick
                for entry in reversed(self.recent_logs):
                    if (entry.pick_id == log_entry.pick_id and 
                        entry.event_type == AgentEvent.PICK_GENERATED):
                        
                        agent_key = f"{entry.agent_name}:{entry.sport}"
                        if agent_key in self.performance_cache['agent_performance']:
                            agent_perf = self.performance_cache['agent_performance'][agent_key]
                            
                            if log_entry.outcome == 'win':
                                agent_perf['picks_won'] += 1
                            elif log_entry.outcome == 'loss':
                                agent_perf['picks_lost'] += 1
                            
                            if log_entry.profit_loss:
                                agent_perf['total_profit'] += log_entry.profit_loss
                        break
            
            # Update cost tracking
            if log_entry.cost_usd:
                self.performance_cache['total_cost'] += log_entry.cost_usd
            
            # Update timestamp
            self.performance_cache['last_updated'] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_cache.copy()
    
    def get_agent_performance(self, agent_name: str = None, sport: str = None) -> Dict[str, Any]:
        """Get performance metrics for specific agent or all agents"""
        if agent_name and sport:
            agent_key = f"{agent_name}:{sport}"
            return self.performance_cache['agent_performance'].get(agent_key, {})
        else:
            return self.performance_cache['agent_performance'].copy()
    
    def get_session_logs(self, session_id: str) -> List[AgentLogEntry]:
        """Get all logs for a specific session"""
        return self.session_cache.get(session_id, [])
    
    def get_recent_logs(self, limit: int = 100, 
                       event_type: Optional[AgentEvent] = None,
                       agent_name: Optional[str] = None,
                       sport: Optional[str] = None) -> List[AgentLogEntry]:
        """Get recent logs with optional filtering"""
        logs = self.recent_logs[-limit:]
        
        if event_type:
            logs = [log for log in logs if log.event_type == event_type]
        
        if agent_name:
            logs = [log for log in logs if log.agent_name == agent_name]
        
        if sport:
            logs = [log for log in logs if log.sport == sport]
        
        return logs
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff_time = datetime.now(timezone.utc) - pd.Timedelta(hours=hours)
        
        recent_errors = []
        for log in self.recent_logs:
            if (log.level == LogLevel.ERROR and 
                datetime.fromisoformat(log.timestamp.replace('Z', '+00:00')) > cutoff_time):
                recent_errors.append(log)
        
        error_summary = {
            'total_errors': len(recent_errors),
            'error_types': {},
            'affected_agents': set(),
            'error_rate': 0.0
        }
        
        for error in recent_errors:
            error_type = error.error_type or 'Unknown'
            if error_type not in error_summary['error_types']:
                error_summary['error_types'][error_type] = 0
            error_summary['error_types'][error_type] += 1
            error_summary['affected_agents'].add(f"{error.agent_name}:{error.sport}")
        
        error_summary['affected_agents'] = list(error_summary['affected_agents'])
        
        # Calculate error rate
        total_requests = self.performance_cache['requests_total']
        if total_requests > 0:
            error_summary['error_rate'] = len(recent_errors) / total_requests
        
        return error_summary
    
    def export_logs(self, 
                   output_file: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   format: str = 'json') -> str:
        """Export logs to file with optional date filtering"""
        try:
            # Read all log files
            all_logs = []
            for log_file in sorted(self.log_dir.glob("agent_logs_*.jsonl")):
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            log_data = json.loads(line.strip())
                            all_logs.append(log_data)
                        except json.JSONDecodeError:
                            continue
            
            # Apply date filtering
            if start_date or end_date:
                filtered_logs = []
                for log in all_logs:
                    log_time = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                    
                    if start_date:
                        start_time = datetime.fromisoformat(start_date)
                        if log_time < start_time:
                            continue
                    
                    if end_date:
                        end_time = datetime.fromisoformat(end_date)
                        if log_time > end_time:
                            continue
                    
                    filtered_logs.append(log)
                all_logs = filtered_logs
            
            # Export in requested format
            output_path = Path(output_file)
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(all_logs, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                if all_logs:
                    df = pd.json_normalize(all_logs)
                    df.to_csv(output_path, index=False)
                else:
                    # Create empty CSV with headers
                    pd.DataFrame().to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return str(output_path.absolute())
        
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            raise
    
    def close_session(self, session_id: str):
        """Close and cleanup a logging session"""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info['end_time'] = datetime.now(timezone.utc)
            session_info['duration_seconds'] = (
                session_info['end_time'] - session_info['start_time']
            ).total_seconds()
            
            # Log session summary
            self.log_event(
                event_type=AgentEvent.PERFORMANCE_METRIC,
                level=LogLevel.INFO,
                agent_name=session_info['agent_name'],
                sport=session_info['sport'],
                session_id=session_id,
                metadata={
                    'session_summary': session_info,
                    'total_logs': len(self.session_cache.get(session_id, []))
                }
            )
            
            # Cleanup
            del self.active_sessions[session_id]
            # Keep session cache for a while for analytics
            # Could implement cleanup later if memory becomes an issue

# Global logger instance
_global_logger: Optional[AgentLogger] = None

def get_logger() -> AgentLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger()
    return _global_logger

def init_logger(log_dir: str = "logs", 
               log_level: LogLevel = LogLevel.INFO,
               **kwargs) -> AgentLogger:
    """Initialize the global logger with custom settings"""
    global _global_logger
    _global_logger = AgentLogger(log_dir=log_dir, log_level=log_level, **kwargs)
    return _global_logger