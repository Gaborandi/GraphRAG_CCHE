# services/monitoring.py
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import threading
import time
import json
import os
from collections import defaultdict
import psutil
import numpy as np
from dataclasses import dataclass
import traceback

from ..config import Config
from ..database.graph import Neo4jConnection

@dataclass
class MetricPoint:
    """Container for a single metric measurement."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]

class MetricCollector:
    """Collects and manages metrics."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.values: List[MetricPoint] = []
        self._lock = threading.Lock()

    def add_value(self, value: float, labels: Dict[str, str] = None):
        """Add a new metric value."""
        with self._lock:
            self.values.append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            ))
            
            # Keep only last 24 hours of data
            cutoff = datetime.now() - timedelta(days=1)
            self.values = [v for v in self.values if v.timestamp > cutoff]

    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of metric values."""
        with self._lock:
            if not self.values:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'std': 0.0
                }
            
            values = [v.value for v in self.values]
            return {
                'count': len(values),
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }

class SystemMonitor:
    """Monitors system resource usage."""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': MetricCollector(
                'cpu_usage',
                'CPU usage percentage'
            ),
            'memory_usage': MetricCollector(
                'memory_usage',
                'Memory usage percentage'
            ),
            'disk_usage': MetricCollector(
                'disk_usage',
                'Disk usage percentage'
            ),
            'process_count': MetricCollector(
                'process_count',
                'Number of running processes'
            )
        }
        
    def collect_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].add_value(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].add_value(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['disk_usage'].add_value(disk.percent)
            
            # Process count
            process_count = len(psutil.pids())
            self.metrics['process_count'].add_value(process_count)
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {str(e)}")

class DatabaseMonitor:
    """Monitors Neo4j database metrics."""
    
    def __init__(self, neo4j_connection: Neo4jConnection):
        self.neo4j = neo4j_connection
        self.metrics = {
            'active_transactions': MetricCollector(
                'active_transactions',
                'Number of active transactions'
            ),
            'query_time': MetricCollector(
                'query_time',
                'Average query execution time'
            ),
            'cache_hits': MetricCollector(
                'cache_hits',
                'Cache hit ratio'
            ),
            'memory_usage': MetricCollector(
                'memory_usage',
                'Database memory usage'
            )
        }

    def collect_metrics(self):
        """Collect current database metrics."""
        try:
            with self.neo4j._driver.session() as session:
                # Query metrics
                result = session.run("""
                CALL dbms.queryJmx('org.neo4j:*')
                YIELD name, attributes
                RETURN name, attributes
                """)
                
                for record in result:
                    name = record['name']
                    attrs = record['attributes']
                    
                    if 'Transactions' in name:
                        self.metrics['active_transactions'].add_value(
                            attrs.get('NumberOfOpenTransactions', 0)
                        )
                    elif 'Cache' in name:
                        self.metrics['cache_hits'].add_value(
                            attrs.get('HitRatio', 0.0)
                        )
                    elif 'Memory' in name:
                        self.metrics['memory_usage'].add_value(
                            attrs.get('UsedHeapMemory', 0)
                        )
                        
        except Exception as e:
            logging.error(f"Error collecting database metrics: {str(e)}")

class OperationMonitor:
    """Monitors graph operations."""
    
    def __init__(self):
        self.metrics = {
            'operation_count': MetricCollector(
                'operation_count',
                'Number of operations'
            ),
            'operation_time': MetricCollector(
                'operation_time',
                'Operation execution time'
            ),
            'error_rate': MetricCollector(
                'error_rate',
                'Operation error rate'
            ),
            'success_rate': MetricCollector(
                'success_rate',
                'Operation success rate'
            )
        }
        
        self.operation_history = defaultdict(list)
        self._history_lock = threading.Lock()

    def record_operation(self, operation_type: str, 
                        duration: float, success: bool):
        """Record an operation execution."""
        try:
            with self._history_lock:
                self.operation_history[operation_type].append({
                    'timestamp': datetime.now(),
                    'duration': duration,
                    'success': success
                })
                
                # Update metrics
                self.metrics['operation_count'].add_value(
                    1,
                    {'type': operation_type}
                )
                self.metrics['operation_time'].add_value(
                    duration,
                    {'type': operation_type}
                )
                
                # Calculate error and success rates
                ops = self.operation_history[operation_type]
                recent_ops = [
                    op for op in ops
                    if op['timestamp'] > datetime.now() - timedelta(hours=1)
                ]
                
                if recent_ops:
                    success_rate = sum(
                        1 for op in recent_ops if op['success']
                    ) / len(recent_ops)
                    self.metrics['success_rate'].add_value(
                        success_rate,
                        {'type': operation_type}
                    )
                    self.metrics['error_rate'].add_value(
                        1.0 - success_rate,
                        {'type': operation_type}
                    )
                    
                # Cleanup old history
                cutoff = datetime.now() - timedelta(days=1)
                self.operation_history[operation_type] = [
                    op for op in ops if op['timestamp'] > cutoff
                ]
                
        except Exception as e:
            logging.error(f"Error recording operation: {str(e)}")

class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'response_time': MetricCollector(
                'response_time',
                'API response time'
            ),
            'throughput': MetricCollector(
                'throughput',
                'Requests per second'
            ),
            'queue_length': MetricCollector(
                'queue_length',
                'Request queue length'
            ),
            'error_count': MetricCollector(
                'error_count',
                'Number of errors'
            )
        }
        
        self.request_history = []
        self._history_lock = threading.Lock()

    def record_request(self, duration: float, success: bool,
                      endpoint: str):
        """Record an API request."""
        try:
            with self._history_lock:
                self.request_history.append({
                    'timestamp': datetime.now(),
                    'duration': duration,
                    'success': success,
                    'endpoint': endpoint
                })
                
                # Update metrics
                self.metrics['response_time'].add_value(
                    duration,
                    {'endpoint': endpoint}
                )
                
                if not success:
                    self.metrics['error_count'].add_value(
                        1,
                        {'endpoint': endpoint}
                    )
                    
                # Calculate throughput
                recent_requests = [
                    req for req in self.request_history
                    if req['timestamp'] > datetime.now() - timedelta(minutes=1)
                ]
                self.metrics['throughput'].add_value(
                    len(recent_requests),
                    {'endpoint': endpoint}
                )
                
                # Cleanup old history
                cutoff = datetime.now() - timedelta(hours=1)
                self.request_history = [
                    req for req in self.request_history
                    if req['timestamp'] > cutoff
                ]
                
        except Exception as e:
            logging.error(f"Error recording request: {str(e)}")

class MonitoringService:
    """Main monitoring service that coordinates all monitors."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.system_monitor = SystemMonitor()
        self.database_monitor = DatabaseMonitor(neo4j_connection)
        self.operation_monitor = OperationMonitor()
        self.performance_monitor = PerformanceMonitor()
        
        # Start monitoring thread
        self._start_monitoring()
        
        # Setup logging
        self._setup_logging()

    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return {
            'system': {
                name: collector.get_statistics()
                for name, collector in self.system_monitor.metrics.items()
            },
            'database': {
                name: collector.get_statistics()
                for name, collector in self.database_monitor.metrics.items()
            },
            'operations': {
                name: collector.get_statistics()
                for name, collector in self.operation_monitor.metrics.items()
            },
            'performance': {
                name: collector.get_statistics()
                for name, collector in self.performance_monitor.metrics.items()
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        metrics = self.get_metrics()
        
        # Define thresholds
        thresholds = self.config.model_config.get('monitoring', {}).get(
            'thresholds',
            {
                'cpu_usage': 80.0,
                'memory_usage': 80.0,
                'disk_usage': 80.0,
                'error_rate': 0.1
            }
        )
        
        # Check system health
        status = {
            'healthy': True,
            'warnings': [],
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Check system metrics
        system = metrics['system']
        if system['cpu_usage']['mean'] > thresholds['cpu_usage']:
            status['warnings'].append('High CPU usage')
        if system['memory_usage']['mean'] > thresholds['memory_usage']:
            status['warnings'].append('High memory usage')
        if system['disk_usage']['mean'] > thresholds['disk_usage']:
            status['warnings'].append('High disk usage')
            
        # Check error rates
        operations = metrics['operations']
        if operations['error_rate']['mean'] > thresholds['error_rate']:
            status['errors'].append('High error rate')
            
        # Update overall health status
        if status['errors']:
            status['healthy'] = False
            
        return status

    def _start_monitoring(self):
        """Start monitoring thread."""
        def monitoring_task():
            while True:
                try:
                    # Collect metrics
                    self.system_monitor.collect_metrics()
                    self.database_monitor.collect_metrics()
                    
                    # Check health status
                    health = self.get_health_status()
                    if not health['healthy']:
                        logging.warning(
                            f"System health issues detected: {health['warnings'] + health['errors']}"
                        )
                        
                    time.sleep(60)  # Collect metrics every minute
                    
                except Exception as e:
                    logging.error(f"Error in monitoring task: {str(e)}")

        thread = threading.Thread(target=monitoring_task, daemon=True)
        thread.start()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.model_config.get('logging', {})
        
        logging.basicConfig(
            level=log_config.get('level', 'INFO'),
            format=log_config.get(
                'format',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            filename=log_config.get('file', 'graphrag.log'),
            filemode='a'
        )
        
        # Also log to console if specified
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_config.get('console_level', 'INFO'))
            formatter = logging.Formatter(log_config.get(
                'console_format',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)