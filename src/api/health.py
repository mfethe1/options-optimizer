"""
Enhanced Health Check

Provides detailed health status for all system components.
"""

from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def check_database_health() -> Dict[str, Any]:
    """
    Check database health.
    
    Returns:
        Dictionary with status and details
    """
    try:
        # For now, we're using in-memory storage
        # In production, this would check actual database connection
        return {
            "status": "healthy",
            "type": "in-memory",
            "message": "Database is operational"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "type": "in-memory",
            "message": str(e)
        }


def check_swarm_health() -> Dict[str, Any]:
    """
    Check swarm coordinator health.
    
    Returns:
        Dictionary with status and details
    """
    try:
        # Check if swarm coordinator is available
        from src.agents.swarm import SwarmCoordinator
        
        return {
            "status": "healthy",
            "message": "Swarm coordinator is available",
            "agents": 8
        }
    except Exception as e:
        logger.error(f"Swarm health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


def check_auth_health() -> Dict[str, Any]:
    """
    Check authentication system health.
    
    Returns:
        Dictionary with status and details
    """
    try:
        from src.data.user_store import user_store
        
        user_count = len(user_store.list_users())
        
        return {
            "status": "healthy",
            "message": "Authentication system is operational",
            "users": user_count
        }
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


def check_monitoring_health() -> Dict[str, Any]:
    """
    Check monitoring system health.
    
    Returns:
        Dictionary with status and details
    """
    try:
        import sentry_sdk
        from prometheus_client import REGISTRY
        
        # Check if Sentry is configured
        sentry_configured = sentry_sdk.Hub.current.client is not None
        
        # Count Prometheus metrics
        metric_count = len(list(REGISTRY.collect()))
        
        return {
            "status": "healthy",
            "sentry_enabled": sentry_configured,
            "prometheus_metrics": metric_count,
            "message": "Monitoring systems are operational"
        }
    except Exception as e:
        logger.error(f"Monitoring health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


def get_detailed_health() -> Dict[str, Any]:
    """
    Get detailed health status for all components.
    
    Returns:
        Dictionary with overall status and component details
    """
    components = {
        "database": check_database_health(),
        "swarm": check_swarm_health(),
        "authentication": check_auth_health(),
        "monitoring": check_monitoring_health()
    }
    
    # Determine overall status
    all_healthy = all(
        component["status"] == "healthy"
        for component in components.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": components
    }

