"""Background task service for periodic operations."""

import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
import time
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.logging import get_logger


class BackgroundTaskService:
    """Service for managing background periodic tasks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        
    def add_task(
        self, 
        name: str, 
        coro_func: Callable, 
        interval_hours: float, 
        run_immediately: bool = False
    ):
        """Add a periodic task.
        
        Args:
            name: Unique task name
            coro_func: Async function to execute
            interval_hours: Interval between executions in hours
            run_immediately: Whether to run the task immediately on start
        """
        self.tasks[name] = {
            'coro_func': coro_func,
            'interval_seconds': interval_hours * 3600,
            'run_immediately': run_immediately,
            'last_run': None,
            'next_run': None,
            'running': False,
            'error_count': 0,
            'success_count': 0
        }
        self.logger.info(f"Added background task '{name}' with {interval_hours}h interval")
    
    def start(self):
        """Start the background task service."""
        if self.running:
            self.logger.warning("Background task service is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        self.logger.info("Background task service started")
    
    def stop(self):
        """Stop the background task service."""
        if not self.running:
            return
        
        self.running = False
        if self.loop and not self.loop.is_closed():
            # Schedule the shutdown in the event loop
            asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        self.logger.info("Background task service stopped")
    
    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Schedule all tasks
            for name, task_info in self.tasks.items():
                self.loop.create_task(self._run_task(name, task_info))
            
            # Run the event loop
            self.loop.run_forever()
        except Exception as e:
            self.logger.error(f"Error in background task event loop: {e}")
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()
    
    async def _shutdown(self):
        """Shutdown the event loop gracefully."""
        # Cancel all running tasks
        tasks = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
        if tasks:
            self.logger.info(f"Cancelling {len(tasks)} running tasks")
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop the event loop
        self.loop.stop()
    
    async def _run_task(self, name: str, task_info: Dict[str, Any]):
        """Run a single periodic task."""
        coro_func = task_info['coro_func']
        interval_seconds = task_info['interval_seconds']
        run_immediately = task_info['run_immediately']
        
        # Calculate initial delay
        if run_immediately:
            initial_delay = 0
        else:
            initial_delay = interval_seconds
        
        # Set next run time
        task_info['next_run'] = datetime.now() + timedelta(seconds=initial_delay)
        
        self.logger.info(f"Scheduled task '{name}' - next run: {task_info['next_run']}")
        
        # Initial delay
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        
        while self.running:
            try:
                task_info['running'] = True
                task_info['last_run'] = datetime.now()
                
                self.logger.info(f"Executing background task: {name}")
                start_time = time.time()
                
                # Execute the task
                await coro_func()
                
                execution_time = time.time() - start_time
                task_info['success_count'] += 1
                
                self.logger.info(
                    f"Task '{name}' completed successfully in {execution_time:.2f}s "
                    f"(success: {task_info['success_count']}, errors: {task_info['error_count']})"
                )
                
            except asyncio.CancelledError:
                self.logger.info(f"Task '{name}' was cancelled")
                break
            except Exception as e:
                task_info['error_count'] += 1
                self.logger.error(
                    f"Error in background task '{name}': {e} "
                    f"(success: {task_info['success_count']}, errors: {task_info['error_count']})"
                )
            finally:
                task_info['running'] = False
                task_info['next_run'] = datetime.now() + timedelta(seconds=interval_seconds)
            
            # Wait for next execution
            if self.running:
                self.logger.debug(f"Task '{name}' sleeping for {interval_seconds}s until {task_info['next_run']}")
                await asyncio.sleep(interval_seconds)
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all background tasks."""
        status = {
            'service_running': self.running,
            'tasks': {}
        }
        
        for name, task_info in self.tasks.items():
            status['tasks'][name] = {
                'interval_hours': task_info['interval_seconds'] / 3600,
                'last_run': task_info['last_run'].isoformat() if task_info['last_run'] else None,
                'next_run': task_info['next_run'].isoformat() if task_info['next_run'] else None,
                'currently_running': task_info['running'],
                'success_count': task_info['success_count'],
                'error_count': task_info['error_count']
            }
        
        return status
    
    def force_run_task(self, task_name: str) -> bool:
        """Force immediate execution of a specific task."""
        if not self.running or not self.loop:
            self.logger.error("Background task service is not running")
            return False
        
        if task_name not in self.tasks:
            self.logger.error(f"Task '{task_name}' not found")
            return False
        
        task_info = self.tasks[task_name]
        if task_info['running']:
            self.logger.warning(f"Task '{task_name}' is already running")
            return False
        
        # Schedule immediate execution
        asyncio.run_coroutine_threadsafe(
            self._execute_task_once(task_name, task_info), 
            self.loop
        )
        
        self.logger.info(f"Scheduled immediate execution of task '{task_name}'")
        return True
    
    async def _execute_task_once(self, name: str, task_info: Dict[str, Any]):
        """Execute a task once immediately."""
        try:
            task_info['running'] = True
            self.logger.info(f"Force executing background task: {name}")
            
            start_time = time.time()
            await task_info['coro_func']()
            execution_time = time.time() - start_time
            
            task_info['success_count'] += 1
            task_info['last_run'] = datetime.now()
            
            self.logger.info(f"Force execution of '{name}' completed in {execution_time:.2f}s")
            
        except Exception as e:
            task_info['error_count'] += 1
            self.logger.error(f"Error in force execution of '{name}': {e}")
        finally:
            task_info['running'] = False
