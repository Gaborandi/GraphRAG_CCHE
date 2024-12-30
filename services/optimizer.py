# services/optimizer.py
from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime
import threading
from collections import defaultdict
import time
import numpy as np

#from ..config import Config
from config import Config     
from .cache import cache_manager, QueryCache
from database.graph import Neo4jConnection

class PerformanceOptimizer:
    """Handles performance optimization for graph operations."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize query cache
        self.query_cache = QueryCache()
        
        # Performance metrics
        self.metrics = defaultdict(list)
        self._metrics_lock = threading.Lock()
        
        # Load optimization configurations
        self.optimizer_config = config.model_config.get('optimizer', {})
        
        # Start monitoring thread
        self._start_monitoring()

    def optimize_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a query for better performance."""
        try:
            # Check query cache first
            cached_result = self.query_cache.get_query_result(query, params)
            if cached_result is not None:
                return cached_result
            
            # Analyze query
            analysis = self._analyze_query(query)
            
            # Apply optimizations
            optimized_query = query
            optimized_params = params.copy()
            
            if analysis['can_use_index']:
                optimized_query = self._add_index_hints(optimized_query)
            
            if analysis['can_limit_results']:
                optimized_query = self._add_result_limit(
                    optimized_query,
                    self.optimizer_config.get('default_limit', 1000)
                )
            
            # Track optimization
            self._track_optimization(
                'query_optimization',
                {
                    'original_query': query,
                    'optimized_query': optimized_query,
                    'analysis': analysis
                }
            )
            
            return {
                'query': optimized_query,
                'params': optimized_params,
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing query: {str(e)}")
            return {
                'query': query,
                'params': params,
                'error': str(e)
            }

    def optimize_operation(self, operation_type: str,
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a graph operation."""
        try:
            optimized_data = data.copy()
            
            if operation_type == 'create':
                optimized_data = self._optimize_create(optimized_data)
            elif operation_type == 'update':
                optimized_data = self._optimize_update(optimized_data)
            elif operation_type == 'delete':
                optimized_data = self._optimize_delete(optimized_data)
            
            # Track optimization
            self._track_optimization(
                'operation_optimization',
                {
                    'type': operation_type,
                    'original_data': data,
                    'optimized_data': optimized_data
                }
            )
            
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"Error optimizing operation: {str(e)}")
            return data

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics."""
        with self._metrics_lock:
            metrics = {}
            
            for metric_type, values in self.metrics.items():
                if values:
                    metrics[metric_type] = {
                        'count': len(values),
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values)
                    }
            
            return metrics

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query for optimization opportunities."""
        analysis = {
            'can_use_index': False,
            'can_limit_results': False,
            'estimated_cost': 0
        }
        
        # Check if query can use indexes
        if any(hint in query.upper() for hint in ['WHERE', 'ORDER BY', 'RETURN']):
            analysis['can_use_index'] = True
            
        # Check if results can be limited
        if 'RETURN' in query.upper() and 'LIMIT' not in query.upper():
            analysis['can_limit_results'] = True
            
        # Estimate query cost
        analysis['estimated_cost'] = self._estimate_query_cost(query)
        
        return analysis

    def _estimate_query_cost(self, query: str) -> float:
        """Estimate computational cost of a query."""
        cost = 1.0
        
        # Add cost for each operation
        operations = {
            'MATCH': 1.0,
            'WHERE': 0.5,
            'WITH': 0.3,
            'ORDER BY': 2.0,
            'RETURN': 0.1
        }
        
        for op, op_cost in operations.items():
            if op in query.upper():
                cost += op_cost
                
        return cost

    def _optimize_create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize create operation."""
        optimized = data.copy()
        
        # Add any missing indexes
        if 'properties' in optimized:
            self._ensure_indexes(optimized['properties'])
            
        return optimized

    def _optimize_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize update operation."""
        optimized = data.copy()
        
        # Optimize property updates
        if 'properties' in optimized:
            optimized['properties'] = self._optimize_properties(
                optimized['properties']
            )
            
        return optimized

    def _optimize_delete(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize delete operation."""
        optimized = data.copy()
        
        # Add cascade delete if needed
        if self.optimizer_config.get('cascade_delete', False):
            optimized['cascade'] = True
            
        return optimized

    def _optimize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize property updates."""
        optimized = {}
        
        # Only include changed properties
        for key, value in properties.items():
            if self._is_property_changed(key, value):
                optimized[key] = value
                
        return optimized

    def _ensure_indexes(self, properties: Dict[str, Any]):
        """Ensure indexes exist for frequently accessed properties."""
        indexed_properties = self._get_indexed_properties()
        
        for key in properties:
            if (key not in indexed_properties and 
                self._should_create_index(key)):
                self._create_index(key)

    def _is_property_changed(self, key: str, value: Any) -> bool:
        """Check if property value has changed."""
        # Implementation depends on change tracking mechanism
        return True

    def _should_create_index(self, property_name: str) -> bool:
        """Determine if index should be created for property."""
        # Check access patterns and configuration
        return False

    def _create_index(self, property_name: str):
        """Create index for property."""
        query = f"""
        CREATE INDEX ON :Entity({property_name})
        """
        
        with self.neo4j._driver.session() as session:
            session.run(query)

    def _get_indexed_properties(self) -> Set[str]:
        """Get currently indexed properties."""
        query = """
        CALL db.indexes()
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            indexes = set()
            for record in result:
                # Extract property names from index information
                if 'properties' in record:
                    indexes.update(record['properties'])
            return indexes

    def _add_index_hints(self, query: str) -> str:
        """Add index usage hints to query."""
        # Add USING INDEX hints where appropriate
        return query

    def _add_result_limit(self, query: str, limit: int) -> str:
        """Add result limit to query."""
        if 'LIMIT' not in query.upper():
            query = f"{query.rstrip()} LIMIT {limit}"
        return query

    def _track_optimization(self, optimization_type: str,
                          details: Dict[str, Any]):
        """Track optimization for metrics."""
        with self._metrics_lock:
            self.metrics[optimization_type].append({
                'timestamp': datetime.now(),
                'details': details
            })

    def _start_monitoring(self):
        """Start performance monitoring thread."""
        def monitoring_task():
            while True:
                try:
                    self._collect_metrics()
                    time.sleep(60)  # Collect metrics every minute
                except Exception as e:
                    self.logger.error(f"Monitoring error: {str(e)}")

        thread = threading.Thread(
            target=monitoring_task,
            daemon=True
        )
        thread.start()

    def _collect_metrics(self):
        """Collect performance metrics."""
        try:
            # Collect query metrics
            self._collect_query_metrics()
            
            # Collect cache metrics
            self._collect_cache_metrics()
            
            # Collect database metrics
            self._collect_database_metrics()
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")

    def _collect_query_metrics(self):
        """Collect query performance metrics."""
        metrics = {
            'cache_hit_rate': self.query_cache.stats['hits'] / 
                            (self.query_cache.stats['hits'] + 
                             self.query_cache.stats['misses'])
            if (self.query_cache.stats['hits'] + 
                self.query_cache.stats['misses']) > 0 else 0
        }
        
        with self._metrics_lock:
            self.metrics['query_performance'].append(metrics)

    def _collect_cache_metrics(self):
        """Collect cache performance metrics."""
        metrics = cache_manager.get_stats()
        
        with self._metrics_lock:
            self.metrics['cache_performance'].append(metrics)

    def _collect_database_metrics(self):
        """Collect database performance metrics."""
        query = """
        CALL dbms.queryJmx('org.neo4j:*')
        """
        
        try:
            with self.neo4j._driver.session() as session:
                result = session.run(query)
                metrics = {}
                
                for record in result:
                    if 'name' in record and 'attributes' in record:
                        name = record['name']
                        attrs = record['attributes']
                        
                        if 'Memory' in name:
                            metrics['memory_usage'] = attrs.get('UsedHeapMemory', 0)
                        elif 'Transactions' in name:
                            metrics['active_transactions'] = attrs.get(
                                'NumberOfOpenTransactions',
                                0
                            )
                
                with self._metrics_lock:
                    self.metrics['database_performance'].append(metrics)
                    
        except Exception as e:
            self.logger.error(f"Error collecting database metrics: {str(e)}")

class QueryOptimizationContext:
    """Context manager for query optimization."""
    
    def __init__(self, optimizer: PerformanceOptimizer,
                 query: str, params: Dict[str, Any]):
        self.optimizer = optimizer
        self.original_query = query
        self.original_params = params
        self.start_time = None
        self.optimized_query = None
        self.optimized_params = None

    def __enter__(self):
        self.start_time = time.time()
        optimization = self.optimizer.optimize_query(
            self.original_query,
            self.original_params
        )
        self.optimized_query = optimization['query']
        self.optimized_params = optimization['params']
        return self.optimized_query, self.optimized_params

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        # Track query execution
        self.optimizer._track_optimization(
            'query_execution',
            {
                'original_query': self.original_query,
                'optimized_query': self.optimized_query,
                'duration': duration,
                'error': str(exc_val) if exc_val else None
            }
        )