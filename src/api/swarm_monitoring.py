"""
Swarm Analysis Monitoring and Diagnostics

Provides comprehensive logging, progress tracking, and diagnostic capabilities
for troubleshooting and monitoring the 17-agent swarm system.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import threading
import uuid

logger = logging.getLogger(__name__)


class AnalysisMonitor:
    """
    Monitors swarm analysis progress and provides real-time status updates.
    Thread-safe singleton for tracking analysis across requests.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._lock = threading.RLock()
        
        # Active analyses
        self._active_analyses: Dict[str, Dict[str, Any]] = {}
        
        # Historical data
        self._completed_analyses: List[Dict[str, Any]] = []
        self._max_history = 100
        
        # Agent performance tracking
        self._agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'last_error': None,
            'last_success': None
        })
        
        logger.info("AnalysisMonitor initialized")
    
    def start_analysis(self, metadata: Dict[str, Any]) -> str:
        """Start tracking a new analysis, returns analysis_id"""
        analysis_id = str(uuid.uuid4())[:8]
        
        with self._lock:
            self._active_analyses[analysis_id] = {
                'id': analysis_id,
                'status': 'initializing',
                'start_time': time.time(),
                'current_step': 'Starting analysis...',
                'agent_progress': {},
                'metadata': metadata,
                'errors': [],
                'warnings': [],
                'metrics': {}
            }
            logger.info(f"🚀 Started tracking analysis: {analysis_id}")
        
        return analysis_id
    
    def update_step(self, analysis_id: str, step: str, details: Optional[str] = None) -> None:
        """Update the current step of an analysis"""
        with self._lock:
            if analysis_id in self._active_analyses:
                self._active_analyses[analysis_id]['current_step'] = step
                if details:
                    self._active_analyses[analysis_id]['step_details'] = details
                logger.info(f"[{analysis_id}] ▶ {step}")
    
    def update_agent_progress(
        self,
        analysis_id: str,
        agent_id: str,
        status: str,
        message: Optional[str] = None,
        progress: Optional[float] = None
    ) -> None:
        """Update progress for a specific agent"""
        with self._lock:
            if analysis_id in self._active_analyses:
                self._active_analyses[analysis_id]['agent_progress'][agent_id] = {
                    'agent_id': agent_id,
                    'status': status,
                    'message': message,
                    'progress': progress,
                    'timestamp': datetime.utcnow().isoformat()
                }
                logger.debug(f"[{analysis_id}] 🤖 {agent_id}: {status}")
    
    def record_agent_call(
        self,
        agent_id: str,
        success: bool,
        duration: float,
        error: Optional[str] = None
    ) -> None:
        """Record an agent call for performance tracking"""
        with self._lock:
            stats = self._agent_stats[agent_id]
            stats['total_calls'] += 1
            
            if success:
                stats['successful_calls'] += 1
                stats['last_success'] = datetime.utcnow().isoformat()
            else:
                stats['failed_calls'] += 1
                stats['last_error'] = error
            
            stats['total_time'] += duration
            stats['avg_time'] = stats['total_time'] / stats['total_calls']
    
    def add_error(self, analysis_id: str, error: str) -> None:
        """Add an error to the analysis"""
        with self._lock:
            if analysis_id in self._active_analyses:
                self._active_analyses[analysis_id]['errors'].append({
                    'message': error,
                    'timestamp': datetime.utcnow().isoformat()
                })
                logger.error(f"[{analysis_id}] ✗ {error}")
    
    def add_warning(self, analysis_id: str, warning: str) -> None:
        """Add a warning to the analysis"""
        with self._lock:
            if analysis_id in self._active_analyses:
                self._active_analyses[analysis_id]['warnings'].append({
                    'message': warning,
                    'timestamp': datetime.utcnow().isoformat()
                })
                logger.warning(f"[{analysis_id}] ⚠ {warning}")
    
    def add_metric(self, analysis_id: str, metric_name: str, value: Any) -> None:
        """Add a metric to the analysis"""
        with self._lock:
            if analysis_id in self._active_analyses:
                self._active_analyses[analysis_id]['metrics'][metric_name] = value
                logger.info(f"[{analysis_id}] 📊 {metric_name}: {value}")
    
    def complete_analysis(
        self,
        analysis_id: str,
        status: str = 'completed',
        result_summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark an analysis as complete"""
        with self._lock:
            if analysis_id in self._active_analyses:
                analysis = self._active_analyses[analysis_id]
                analysis['status'] = status
                analysis['end_time'] = time.time()
                analysis['duration'] = analysis['end_time'] - analysis['start_time']
                
                if result_summary:
                    analysis['result_summary'] = result_summary
                
                # Move to history
                self._completed_analyses.append(analysis)
                if len(self._completed_analyses) > self._max_history:
                    self._completed_analyses.pop(0)
                
                # Remove from active
                del self._active_analyses[analysis_id]
                
                logger.info(f"✓ Completed analysis: {analysis_id} ({status}) in {analysis['duration']:.2f}s")
    
    def get_analysis_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific analysis"""
        with self._lock:
            if analysis_id in self._active_analyses:
                return self._active_analyses[analysis_id].copy()
            
            # Check history
            for analysis in reversed(self._completed_analyses):
                if analysis['id'] == analysis_id:
                    return analysis.copy()
            
            return None
    
    def get_all_active_analyses(self) -> List[Dict[str, Any]]:
        """Get all active analyses"""
        with self._lock:
            return [a.copy() for a in self._active_analyses.values()]
    
    def get_agent_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all agents"""
        with self._lock:
            return {k: v.copy() for k, v in self._agent_stats.items()}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        with self._lock:
            total_agents = len(self._agent_stats)
            healthy_agents = sum(
                1 for stats in self._agent_stats.values()
                if stats['successful_calls'] > 0 and
                (stats['failed_calls'] / max(stats['total_calls'], 1)) < 0.1
            )
            
            return {
                'active_analyses': len(self._active_analyses),
                'completed_analyses': len(self._completed_analyses),
                'total_agents_tracked': total_agents,
                'healthy_agents': healthy_agents,
                'health_percentage': (healthy_agents / max(total_agents, 1)) * 100,
                'timestamp': datetime.utcnow().isoformat()
            }


# Global monitor instance
monitor = AnalysisMonitor()


class SwarmLogger:
    """
    Enhanced logger with structured output for swarm analysis.
    Provides hierarchical, color-coded logging for better readability.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.analysis_id: Optional[str] = None
    
    def set_analysis_id(self, analysis_id: str) -> None:
        """Set the current analysis ID for context"""
        self.analysis_id = analysis_id
    
    def _log(self, level: str, emoji: str, message: str, **kwargs) -> None:
        """Internal logging with context"""
        prefix = f"[{self.analysis_id}]" if self.analysis_id else ""
        context = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        full_msg = f"{prefix} {emoji} {message} {context}".strip()
        
        if level == 'info':
            self.logger.info(full_msg)
        elif level == 'warning':
            self.logger.warning(full_msg)
        elif level == 'error':
            self.logger.error(full_msg)
        elif level == 'debug':
            self.logger.debug(full_msg)
    
    def step(self, step_name: str, **kwargs) -> None:
        """Log a major step"""
        self._log('info', '▶', step_name, **kwargs)
        if self.analysis_id:
            monitor.update_step(self.analysis_id, step_name)
    
    def agent_start(self, agent_id: str, **kwargs) -> None:
        """Log agent starting"""
        self._log('info', '🤖', f"{agent_id} starting analysis", **kwargs)
        if self.analysis_id:
            monitor.update_agent_progress(self.analysis_id, agent_id, 'analyzing')
    
    def agent_complete(self, agent_id: str, duration: float, **kwargs) -> None:
        """Log agent completion"""
        self._log('info', '✓', f"{agent_id} complete ({duration:.2f}s)", **kwargs)
        if self.analysis_id:
            monitor.update_agent_progress(self.analysis_id, agent_id, 'complete')
            monitor.record_agent_call(agent_id, True, duration)
    
    def agent_error(self, agent_id: str, error: str, **kwargs) -> None:
        """Log agent error"""
        self._log('error', '✗', f"{agent_id} failed: {error}", **kwargs)
        if self.analysis_id:
            monitor.update_agent_progress(self.analysis_id, agent_id, 'failed', message=error)
            monitor.record_agent_call(agent_id, False, 0.0, error)
            monitor.add_error(self.analysis_id, f"{agent_id}: {error}")
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning"""
        self._log('warning', '⚠', message, **kwargs)
        if self.analysis_id:
            monitor.add_warning(self.analysis_id, message)
    
    def success(self, message: str, **kwargs) -> None:
        """Log success"""
        self._log('info', '✓', message, **kwargs)
    
    def metric(self, metric_name: str, value: Any, **kwargs) -> None:
        """Log metric"""
        self._log('info', '📊', f"{metric_name}: {value}", **kwargs)
        if self.analysis_id:
            monitor.add_metric(self.analysis_id, metric_name, value)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info"""
        self._log('info', 'ℹ', message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error"""
        self._log('error', '✗', message, **kwargs)
        if self.analysis_id:
            monitor.add_error(self.analysis_id, message)

