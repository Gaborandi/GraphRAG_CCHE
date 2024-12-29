# services/query_optimization.py
from typing import Dict, Any, List, Optional, Set, Union, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import threading
from collections import defaultdict
import re
import json
import time
import numpy as np
from enum import Enum
import asyncio

from ..config import Config
from ..database.graph import Neo4jConnection
from .cache import QueryCache

class QueryType(Enum):
    """Types of supported queries."""
    ENTITY_SEARCH = "entity_search"
    PATH_FINDING = "path_finding"
    PATTERN_MATCHING = "pattern_matching"
    AGGREGATION = "aggregation"
    SUBGRAPH = "subgraph"
    CUSTOM = "custom"

@dataclass
class QueryStats:
    """Statistics for query execution."""
    execution_time: float
    nodes_accessed: int
    cache_hits: int
    index_hits: int
    memory_usage: float
    optimization_time: float

@dataclass
class QueryPlan:
    """Execution plan for a query."""
    steps: List[Dict[str, Any]]
    estimated_cost: float
    cache_strategy: str
    parallel_steps: List[int]
    index_usage: List[str]

class QueryParser:
    """Parses and normalizes queries."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize query patterns
        self._init_patterns()

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query into structured format."""
        try:
            # Clean and normalize query
            cleaned_query = self._clean_query(query)
            
            # Identify query type
            query_type = self._identify_query_type(cleaned_query)
            
            # Extract query components
            components = self._extract_components(cleaned_query, query_type)
            
            return {
                'type': query_type,
                'components': components,
                'original_query': query,
                'normalized_query': cleaned_query
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing query: {str(e)}")
            raise

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text."""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Normalize keywords
        query = query.upper()
        
        return query

    def _identify_query_type(self, query: str) -> QueryType:
        """Identify type of query."""
        if re.search(r'MATCH.*WHERE', query):
            return QueryType.PATTERN_MATCHING
        elif re.search(r'PATH|SHORTEST', query):
            return QueryType.PATH_FINDING
        elif re.search(r'COUNT|SUM|AVG', query):
            return QueryType.AGGREGATION
        elif re.search(r'SUBGRAPH', query):
            return QueryType.SUBGRAPH
        elif re.search(r'MATCH.*RETURN', query):
            return QueryType.ENTITY_SEARCH
        else:
            return QueryType.CUSTOM

    def _extract_components(self, query: str, 
                          query_type: QueryType) -> Dict[str, Any]:
        """Extract components from query."""
        components = {
            'patterns': [],
            'conditions': [],
            'return_items': [],
            'parameters': {}
        }
        
        # Extract MATCH patterns
        patterns = re.findall(r'MATCH\s+(.*?)(?:WHERE|RETURN|$)', query)
        components['patterns'] = [p.strip() for p in patterns]
        
        # Extract WHERE conditions
        conditions = re.findall(r'WHERE\s+(.*?)(?:RETURN|$)', query)
        components['conditions'] = [c.strip() for c in conditions]
        
        # Extract RETURN items
        returns = re.findall(r'RETURN\s+(.*?)(?:$)', query)
        components['return_items'] = [r.strip() for r in returns]
        
        # Extract parameters
        components['parameters'] = self._extract_parameters(query)
        
        return components

    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameter values from query."""
        params = {}
        
        # Extract numeric parameters
        numeric_params = re.findall(r'\$(\w+)\s*[:=]\s*(\d+(?:\.\d+)?)', query)
        for name, value in numeric_params:
            params[name] = float(value) if '.' in value else int(value)
        
        # Extract string parameters
        string_params = re.findall(r'\$(\w+)\s*[:=]\s*[\'"]([^\'"]+)[\'"]', query)
        params.update(dict(string_params))
        
        return params

    def _init_patterns(self):
        """Initialize regex patterns for query parsing."""
        self.patterns = {
            'entity': r':\w+',
            'relationship': r'\[.*?\]',
            'property': r'\w+\.',
            'function': r'\w+\(',
            'variable': r'\$\w+'
        }

class QueryOptimizer:
    """Optimizes query execution plans."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.parser = QueryParser()
        self.query_cache = QueryCache()
        
        # Load optimization configurations
        self.optimization_config = config.model_config.get('query_optimization', {})
        
        # Initialize statistics tracking
        self.stats = defaultdict(list)
        self._stats_lock = threading.Lock()

    async def optimize_query(self, query: str, 
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize query execution."""
        try:
            start_time = time.time()
            
            # Parse query
            parsed_query = self.parser.parse_query(query)
            
            # Check cache
            cache_key = self._generate_cache_key(parsed_query, params)
            cached_result = self.query_cache.get(cache_key)
            if cached_result:
                return {
                    'result': cached_result,
                    'from_cache': True,
                    'optimization_time': time.time() - start_time
                }
            
            # Generate execution plan
            plan = await self._generate_execution_plan(parsed_query, params)
            
            # Estimate cost
            estimated_cost = self._estimate_execution_cost(plan)
            
            # Optimize plan
            optimized_plan = self._optimize_execution_plan(plan)
            
            optimization_time = time.time() - start_time
            
            return {
                'plan': optimized_plan,
                'estimated_cost': estimated_cost,
                'optimization_time': optimization_time,
                'from_cache': False
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing query: {str(e)}")
            raise

    async def execute_plan(self, plan: QueryPlan) -> Tuple[Any, QueryStats]:
        """Execute optimized query plan."""
        start_time = time.time()
        stats = {
            'nodes_accessed': 0,
            'cache_hits': 0,
            'index_hits': 0,
            'memory_usage': 0.0
        }
        
        try:
            # Execute each step in plan
            results = []
            for step in plan.steps:
                result = await self._execute_step(step, stats)
                results.append(result)
            
            # Combine results
            final_result = self._combine_results(results)
            
            # Calculate execution stats
            execution_time = time.time() - start_time
            query_stats = QueryStats(
                execution_time=execution_time,
                nodes_accessed=stats['nodes_accessed'],
                cache_hits=stats['cache_hits'],
                index_hits=stats['index_hits'],
                memory_usage=stats['memory_usage'],
                optimization_time=plan.optimization_time
            )
            
            return final_result, query_stats
            
        except Exception as e:
            self.logger.error(f"Error executing query plan: {str(e)}")
            raise

    async def _generate_execution_plan(self, parsed_query: Dict[str, Any],
                                     params: Optional[Dict[str, Any]] = None) -> QueryPlan:
        """Generate execution plan for query."""
        query_type = parsed_query['type']
        components = parsed_query['components']
        
        if query_type == QueryType.PATTERN_MATCHING:
            return await self._plan_pattern_matching(components, params)
        elif query_type == QueryType.PATH_FINDING:
            return await self._plan_path_finding(components, params)
        elif query_type == QueryType.AGGREGATION:
            return await self._plan_aggregation(components, params)
        else:
            return await self._plan_default(components, params)

    def _optimize_execution_plan(self, plan: QueryPlan) -> QueryPlan:
        """Optimize execution plan."""
        # Apply various optimization strategies
        plan = self._optimize_step_order(plan)
        plan = self._optimize_cache_usage(plan)
        plan = self._optimize_index_usage(plan)
        
        return plan

    def _optimize_step_order(self, plan: QueryPlan) -> QueryPlan:
        """Optimize order of execution steps."""
        steps = plan.steps.copy()
        
        # Sort steps by estimated cost and dependencies
        dependencies = self._analyze_step_dependencies(steps)
        costs = self._estimate_step_costs(steps)
        
        optimized_steps = self._topological_sort(steps, dependencies, costs)
        plan.steps = optimized_steps
        
        return plan

    def _optimize_cache_usage(self, plan: QueryPlan) -> QueryPlan:
        """Optimize cache usage in plan."""
        for step in plan.steps:
            if step.get('cacheable', False):
                cache_strategy = self._select_cache_strategy(step)
                step['cache_strategy'] = cache_strategy
        
        return plan

    def _optimize_index_usage(self, plan: QueryPlan) -> QueryPlan:
        """Optimize index usage in plan."""
        available_indexes = self._get_available_indexes()
        
        for step in plan.steps:
            if step.get('use_index', False):
                best_index = self._select_best_index(step, available_indexes)
                if best_index:
                    step['index'] = best_index
        
        return plan

    async def _execute_step(self, step: Dict[str, Any],
                          stats: Dict[str, Any]) -> Any:
        """Execute a single step in the plan."""
        operation = step['operation']
        
        # Update stats
        stats['nodes_accessed'] += self._count_nodes_accessed(step)
        if step.get('cache_strategy'):
            stats['cache_hits'] += 1
        if step.get('index'):
            stats['index_hits'] += 1
            
        # Execute appropriate operation
        if operation == 'match_pattern':
            return await self._execute_match_pattern(step)
        elif operation == 'find_paths':
            return await self._execute_find_paths(step)
        elif operation == 'aggregate':
            return await self._execute_aggregate(step)
        else:
            return await self._execute_custom(step)

    def _generate_cache_key(self, parsed_query: Dict[str, Any],
                          params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for query."""
        key_data = {
            'query': parsed_query['normalized_query'],
            'params': params or {}
        }
        return json.dumps(key_data, sort_keys=True)

    def _estimate_execution_cost(self, plan: QueryPlan) -> float:
        """Estimate execution cost of plan."""
        total_cost = 0.0
        
        # Consider various cost factors
        step_costs = self._estimate_step_costs(plan.steps)
        cache_factor = self._get_cache_factor(plan.cache_strategy)
        parallel_factor = self._get_parallel_factor(plan.parallel_steps)
        
        # Calculate total cost
        total_cost = sum(step_costs.values()) * cache_factor * parallel_factor
        
        return total_cost

    def _get_available_indexes(self) -> List[Dict[str, Any]]:
        """Get available indexes from database."""
        try:
            with self.neo4j._driver.session() as session:
                result = session.run("CALL db.indexes()")
                return [dict(record) for record in result]
        except Exception as e:
            self.logger.error(f"Error getting indexes: {str(e)}")
            return []

    def _select_best_index(self, step: Dict[str, Any],
                         available_indexes: List[Dict[str, Any]]) -> Optional[str]:
        """Select best index for step."""
        # Implement index selection logic
        return None

    def _combine_results(self, results: List[Any]) -> Any:
        """Combine results from multiple steps."""
        if not results:
            return None
            
        # Implement result combination logic
        return results[-1]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get query optimization statistics."""
        with self._stats_lock:
            stats = {}
            
            for stat_name, values in self.stats.items():
                if values:
                    stats[stat_name] = {
                        'count': len(values),
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            return stats