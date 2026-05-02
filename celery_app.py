"""
Distributed task queue with Celery for background processing.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure
from celery.result import AsyncResult
import redis
import json

# Configure logging
logger = logging.getLogger("stockvision.celery")

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "stockvision",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["celery_tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1 hour
    task_routes={
        "celery_tasks.ml_prediction": {"queue": "ml"},
        "celery_tasks.sentiment_analysis": {"queue": "sentiment"},
        "celery_tasks.alert_processing": {"queue": "alerts"},
        "celery_tasks.data_fetching": {"queue": "data"},
    },
    task_annotations={
        "*": {"rate_limit": "100/m"},
        "celery_tasks.ml_prediction": {"rate_limit": "10/m"},
        "celery_tasks.sentiment_analysis": {"rate_limit": "20/m"},
    },
    beat_schedule={
        "fetch-market-data": {
            "task": "celery_tasks.fetch_market_data",
            "schedule": 60.0,  # Every minute
        },
        "process-alerts": {
            "task": "celery_tasks.process_alerts",
            "schedule": 30.0,  # Every 30 seconds
        },
        "cleanup-old-results": {
            "task": "celery_tasks.cleanup_old_results",
            "schedule": 3600.0,  # Every hour
        },
    },
)

# Redis client for caching
redis_client = redis.Redis.from_url(CELERY_RESULT_BACKEND, decode_responses=True)

# Custom Task base class for monitoring
class MonitoredTask(Task):
    """Custom task class with monitoring and logging."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"Task {task_id} succeeded: {retval}")
        self._update_task_metrics(task_id, "SUCCESS")
        
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")
        self._update_task_metrics(task_id, "FAILURE", str(exc))
        
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"Task {task_id} retrying: {exc}")
        self._update_task_metrics(task_id, "RETRY")
        
    def _update_task_metrics(self, task_id: str, status: str, error: str = None):
        """Update task metrics in Redis."""
        metrics = {
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "error": error
        }
        
        # Store metrics
        redis_client.hset(f"task_metrics:{task_id}", mapping=metrics)
        redis_client.expire(f"task_metrics:{task_id}", 3600)  # 1 hour
        
        # Update global metrics
        redis_client.lpush("recent_tasks", json.dumps(metrics))
        redis_client.ltrim("recent_tasks", 0, 999)  # Keep last 1000

# Set custom task base class
celery_app.Task = MonitoredTask

# Task monitoring signals
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handler for task pre-run signal."""
    logger.info(f"Starting task {task_id}: {task.name}")
    redis_client.hset(f"task_status:{task_id}", mapping={
        "status": "RUNNING",
        "start_time": datetime.utcnow().isoformat(),
        "task_name": task.name
    })

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handler for task post-run signal."""
    logger.info(f"Completed task {task_id}: {task.name}")
    redis_client.hset(f"task_status:{task_id}", mapping={
        "status": state,
        "end_time": datetime.utcnow().isoformat(),
        "result": str(retval)[:500] if retval else None
    })

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handler for task failure signal."""
    logger.error(f"Task {task_id} failed: {exception}")
    redis_client.hset(f"task_status:{task_id}", mapping={
        "status": "FAILURE",
        "error": str(exception),
        "traceback": str(traceback)[:1000] if traceback else None
    })

# Task management functions
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get detailed task status."""
    # Get basic status from Celery
    result = AsyncResult(task_id, app=celery_app)
    
    # Get detailed status from Redis
    detailed_status = redis_client.hgetall(f"task_status:{task_id}")
    metrics = redis_client.hgetall(f"task_metrics:{task_id}")
    
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.successful() else None,
        "error": str(result.info) if result.failed() else None,
        "progress": result.info.get("progress", 0) if isinstance(result.info, dict) else 0,
        "detailed_status": detailed_status,
        "metrics": metrics,
        "date_done": result.date_done.isoformat() if result.date_done else None
    }

def get_queue_info() -> Dict[str, Any]:
    """Get queue information and statistics."""
    try:
        inspect = celery_app.control.inspect()
        
        # Get active tasks
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        reserved_tasks = inspect.reserved()
        
        # Get worker stats
        stats = inspect.stats()
        
        return {
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "reserved_tasks": reserved_tasks,
            "worker_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting queue info: {e}")
        return {"error": str(e)}

def cancel_task(task_id: str) -> bool:
    """Cancel a running task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        redis_client.hset(f"task_status:{task_id}", "status", "CANCELLED")
        return True
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return False

def get_task_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent task history."""
    try:
        recent_tasks = redis_client.lrange("recent_tasks", 0, limit - 1)
        return [json.loads(task) for task in recent_tasks]
    except Exception as e:
        logger.error(f"Error getting task history: {e}")
        return []

def clear_queue(queue_name: str = None) -> bool:
    """Clear a specific queue or all queues."""
    try:
        if queue_name:
            celery_app.control.purge()
            redis_client.delete(f"celery:{queue_name}")
        else:
            celery_app.control.purge()
        return True
    except Exception as e:
        logger.error(f"Error clearing queue: {e}")
        return False

# Performance monitoring
def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for the task queue."""
    try:
        # Get Redis info
        redis_info = redis_client.info()
        
        # Get task metrics
        task_history = get_task_history(1000)
        
        # Calculate metrics
        total_tasks = len(task_history)
        failed_tasks = len([t for t in task_history if t.get("status") == "FAILURE"])
        success_tasks = len([t for t in task_history if t.get("status") == "SUCCESS"])
        
        # Calculate average execution time (mock implementation)
        avg_execution_time = 45.6  # seconds
        
        return {
            "total_tasks": total_tasks,
            "success_rate": (success_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "failure_rate": (failed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "avg_execution_time": avg_execution_time,
            "redis_memory_usage": redis_info.get("used_memory_human", "N/A"),
            "redis_connected_clients": redis_info.get("connected_clients", 0),
            "celery_workers": len(get_queue_info().get("worker_stats", {})),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {"error": str(e)}

# Health check for Celery
def celery_health_check() -> Dict[str, Any]:
    """Health check for Celery workers."""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if not stats:
            return {"status": "unhealthy", "reason": "No workers available"}
        
        # Check if workers are responding
        ping = inspect.ping()
        
        if not ping:
            return {"status": "unhealthy", "reason": "Workers not responding"}
        
        # Get worker details
        worker_count = len(stats)
        total_tasks = sum(worker.get("total", 0) for worker in stats.values())
        
        return {
            "status": "healthy",
            "worker_count": worker_count,
            "total_tasks_processed": total_tasks,
            "workers": list(stats.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}

# Task scheduling functions
def schedule_task(task_name: str, args: List = None, kwargs: Dict = None, 
                  countdown: int = None, eta: datetime = None, 
                  queue: str = None) -> str:
    """Schedule a task to run at a specific time."""
    try:
        task = celery_app.send_task(
            task_name,
            args=args or [],
            kwargs=kwargs or {},
            countdown=countdown,
            eta=eta,
            queue=queue
        )
        return task.id
    except Exception as e:
        logger.error(f"Error scheduling task {task_name}: {e}")
        raise

def get_scheduled_tasks() -> List[Dict[str, Any]]:
    """Get all scheduled tasks."""
    try:
        inspect = celery_app.control.inspect()
        scheduled = inspect.scheduled()
        
        tasks = []
        for worker, worker_tasks in scheduled.items():
            for task in worker_tasks:
                tasks.append({
                    "worker": worker,
                    "task_id": task.get("request", {}).get("id"),
                    "task_name": task.get("request", {}).get("task"),
                    "eta": task.get("eta"),
                    "args": task.get("request", {}).get("args"),
                    "kwargs": task.get("request", {}).get("kwargs")
                })
        
        return tasks
    except Exception as e:
        logger.error(f"Error getting scheduled tasks: {e}")
        return []

# Batch task operations
def run_batch_tasks(task_name: str, task_list: List[Dict], 
                    max_concurrent: int = 10) -> List[str]:
    """Run multiple tasks concurrently with concurrency limit."""
    task_ids = []
    
    for i, task_data in enumerate(task_list):
        # Control concurrency
        if i > 0 and i % max_concurrent == 0:
            # Wait for some tasks to complete
            check_interval = 5  # seconds
            max_wait = 300  # 5 minutes
            waited = 0
            
            while waited < max_wait:
                active_count = len([t for t in task_ids[-max_concurrent:] 
                                 if get_task_status(t)["state"] in ["PENDING", "STARTED"]])
                if active_count < max_concurrent:
                    break
                import time
                time.sleep(check_interval)
                waited += check_interval
        
        # Schedule next task
        task_id = schedule_task(
            task_name,
            args=task_data.get("args", []),
            kwargs=task_data.get("kwargs", {}),
            queue=task_data.get("queue")
        )
        task_ids.append(task_id)
    
    return task_ids

# Export main app and utilities
__all__ = [
    "celery_app",
    "get_task_status",
    "get_queue_info",
    "cancel_task",
    "get_task_history",
    "get_performance_metrics",
    "celery_health_check",
    "schedule_task",
    "get_scheduled_tasks",
    "run_batch_tasks"
]
