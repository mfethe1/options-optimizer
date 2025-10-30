"""
Monitoring and Diagnostics API Routes

Provides endpoints for monitoring swarm analysis progress,
agent performance, and system health.
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from .swarm_monitoring import monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class AnalysisStatus(BaseModel):
    """Status of a specific analysis"""
    id: str
    status: str
    current_step: str
    start_time: float
    duration: Optional[float] = None
    agent_progress: Dict[str, Any]
    errors: List[Dict[str, str]]
    warnings: List[Dict[str, str]]
    metrics: Dict[str, Any]


class SystemHealth(BaseModel):
    """Overall system health"""
    active_analyses: int
    completed_analyses: int
    total_agents_tracked: int
    healthy_agents: int
    health_percentage: float
    timestamp: str


class AgentStatistics(BaseModel):
    """Performance statistics for an agent"""
    total_calls: int
    successful_calls: int
    failed_calls: int
    avg_time: float
    last_error: Optional[str]
    last_success: Optional[str]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """
    Get overall system health metrics.
    
    Returns:
        - Number of active analyses
        - Number of completed analyses
        - Agent health statistics
        - Overall health percentage
    """
    try:
        health = monitor.get_system_health()
        return health
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/active")
async def get_active_analyses():
    """
    Get all currently active analyses.
    
    Returns:
        List of active analyses with their current status and progress
    """
    try:
        analyses = monitor.get_all_active_analyses()
        return {
            'count': len(analyses),
            'analyses': analyses
        }
    except Exception as e:
        logger.error(f"Error getting active analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Get detailed status of a specific analysis.
    
    Args:
        analysis_id: The analysis ID to query
    
    Returns:
        Detailed status including:
        - Current step
        - Agent progress
        - Errors and warnings
        - Metrics
    """
    try:
        status = monitor.get_analysis_status(analysis_id)
        
        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found"
            )
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/statistics")
async def get_agent_statistics():
    """
    Get performance statistics for all agents.
    
    Returns:
        Statistics for each agent including:
        - Total calls
        - Success/failure rates
        - Average execution time
        - Last error (if any)
    """
    try:
        stats = monitor.get_agent_statistics()
        
        # Calculate summary
        total_calls = sum(s['total_calls'] for s in stats.values())
        total_successes = sum(s['successful_calls'] for s in stats.values())
        total_failures = sum(s['failed_calls'] for s in stats.values())
        
        return {
            'summary': {
                'total_agents': len(stats),
                'total_calls': total_calls,
                'total_successes': total_successes,
                'total_failures': total_failures,
                'overall_success_rate': (total_successes / max(total_calls, 1)) * 100
            },
            'agents': stats
        }
    except Exception as e:
        logger.error(f"Error getting agent statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/statistics")
async def get_specific_agent_statistics(agent_id: str):
    """
    Get performance statistics for a specific agent.
    
    Args:
        agent_id: The agent ID to query
    
    Returns:
        Detailed statistics for the agent
    """
    try:
        all_stats = monitor.get_agent_statistics()
        
        if agent_id not in all_stats:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found or has no statistics"
            )
        
        return {
            'agent_id': agent_id,
            'statistics': all_stats[agent_id]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagnostics")
async def get_diagnostics():
    """
    Get comprehensive diagnostic information for troubleshooting.
    
    Returns:
        - System health
        - Active analyses
        - Agent statistics
        - Recent errors
    """
    try:
        health = monitor.get_system_health()
        active_analyses = monitor.get_all_active_analyses()
        agent_stats = monitor.get_agent_statistics()
        
        # Collect recent errors from active analyses
        recent_errors = []
        for analysis in active_analyses:
            for error in analysis.get('errors', []):
                recent_errors.append({
                    'analysis_id': analysis['id'],
                    'error': error['message'],
                    'timestamp': error['timestamp']
                })
        
        # Find problematic agents
        problematic_agents = []
        for agent_id, stats in agent_stats.items():
            if stats['total_calls'] > 0:
                failure_rate = stats['failed_calls'] / stats['total_calls']
                if failure_rate > 0.1:  # More than 10% failure rate
                    problematic_agents.append({
                        'agent_id': agent_id,
                        'failure_rate': failure_rate * 100,
                        'last_error': stats['last_error']
                    })
        
        return {
            'system_health': health,
            'active_analyses_count': len(active_analyses),
            'recent_errors_count': len(recent_errors),
            'recent_errors': recent_errors[-10:],  # Last 10 errors
            'problematic_agents_count': len(problematic_agents),
            'problematic_agents': problematic_agents,
            'agent_statistics_summary': {
                'total_agents': len(agent_stats),
                'agents_with_calls': sum(1 for s in agent_stats.values() if s['total_calls'] > 0),
                'total_calls': sum(s['total_calls'] for s in agent_stats.values()),
                'total_failures': sum(s['failed_calls'] for s in agent_stats.values())
            }
        }
    except Exception as e:
        logger.error(f"Error getting diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/create-analysis")
async def create_test_analysis():
    """
    Create a test analysis for monitoring system testing.
    
    Returns:
        The created analysis ID
    """
    try:
        analysis_id = monitor.start_analysis({
            'test': True,
            'description': 'Test analysis for monitoring system'
        })
        
        return {
            'analysis_id': analysis_id,
            'message': 'Test analysis created successfully'
        }
    except Exception as e:
        logger.error(f"Error creating test analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def monitoring_info():
    """
    Get information about available monitoring endpoints.
    """
    return {
        'service': 'Swarm Analysis Monitoring',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'System health metrics',
            'GET /analyses/active': 'List active analyses',
            'GET /analyses/{id}': 'Get specific analysis status',
            'GET /agents/statistics': 'Get all agent statistics',
            'GET /agents/{id}/statistics': 'Get specific agent statistics',
            'GET /diagnostics': 'Comprehensive diagnostic information'
        },
        'documentation': '/docs'
    }

