# database/query_optimizer.py
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import torch
from collections import defaultdict

#from ..config import Config
from config import Config
from .graph import Neo4jConnection
from .analytics import GraphAnalyzer
from .community import CommunityDetector

class QueryType(Enum):
    """Types of graph queries."""
    ENTITY_SEARCH = "entity_search"
    PATH_FINDING = "path_finding"
    PATTERN_MATCHING = "pattern_matching"
    AGGREGATION = "aggregation"
    SUBGRAPH = "subgraph"

@dataclass
class QueryPlan:
    """Container for query execution plan."""
    steps: List[Dict[str, Any]]
    estimated_cost: float
    cache_strategy: str
    parallel_steps: List[int]
    index_usage: List[str]

@dataclass
class QueryStats:
    """Container for query execution statistics."""
    execution_time: float
    nodes_accessed: int
    cache_hits: int
    index_hits: int
    memory_usage: float

class QueryOptimizer:
    """Handles query optimization and execution planning."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        self.graph_analyzer = GraphAnalyzer(config, neo4j_connection)
        self.community_detector = CommunityDetector(config, neo4j_connection)
        
        # Cache for statistics and metadata
        self.stats_cache = {}
        self.index_stats = {}
        self.query_history = defaultdict(list)
        
        # Load optimizer configurations
        self.optimizer_config = config.model_config.get('query_optimizer', {})
        
        # Initialize query stats
        self._init_query_stats()

    def optimize_query(self, query: Dict[str, Any]) -> QueryPlan:
        """
        Generate optimized execution plan for a query.
        Args:
            query: Query specification including type, parameters, and constraints
        """
        try:
            # Identify query type
            query_type = self._identify_query_type(query)
            
            # Generate initial plan
            initial_plan = self._generate_initial_plan(query, query_type)
            
            # Apply optimization strategies
            optimized_plan = self._apply_optimizations(initial_plan, query_type)
            
            # Determine cache strategy
            cache_strategy = self._determine_cache_strategy(optimized_plan)
            
            # Identify parallelization opportunities
            parallel_steps = self._identify_parallel_steps(optimized_plan)
            
            # Determine index usage
            index_usage = self._determine_index_usage(optimized_plan)
            
            # Estimate execution cost
            estimated_cost = self._estimate_execution_cost(
                optimized_plan,
                cache_strategy,
                parallel_steps
            )
            
            return QueryPlan(
                steps=optimized_plan,
                estimated_cost=estimated_cost,
                cache_strategy=cache_strategy,
                parallel_steps=parallel_steps,
                index_usage=index_usage
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing query: {str(e)}")
            raise

    def execute_plan(self, plan: QueryPlan) -> Tuple[Any, QueryStats]:
        """Execute an optimized query plan."""
        start_time = datetime.now()
        stats = {
            'nodes_accessed': 0,
            'cache_hits': 0,
            'index_hits': 0,
            'memory_usage': 0.0
        }
        
        try:
            # Execute each step in the plan
            results = []
            for i, step in enumerate(plan.steps):
                # Check if step can be executed in parallel
                if i in plan.parallel_steps:
                    result = self._execute_parallel_step(step, stats)
                else:
                    result = self._execute_step(step, stats)
                results.append(result)
            
            # Combine results based on plan
            final_result = self._combine_results(results, plan)
            
            # Calculate execution stats
            execution_time = (datetime.now() - start_time).total_seconds()
            query_stats = QueryStats(
                execution_time=execution_time,
                nodes_accessed=stats['nodes_accessed'],
                cache_hits=stats['cache_hits'],
                index_hits=stats['index_hits'],
                memory_usage=stats['memory_usage']
            )
            
            # Update query history
            self._update_query_history(plan, query_stats)
            
            return final_result, query_stats
            
        except Exception as e:
            self.logger.error(f"Error executing query plan: {str(e)}")
            raise

    def _identify_query_type(self, query: Dict[str, Any]) -> QueryType:
        """Identify the type of query."""
        if 'path' in query:
            return QueryType.PATH_FINDING
        elif 'pattern' in query:
            return QueryType.PATTERN_MATCHING
        elif 'aggregation' in query:
            return QueryType.AGGREGATION
        elif 'subgraph' in query:
            return QueryType.SUBGRAPH
        else:
            return QueryType.ENTITY_SEARCH

    def _generate_initial_plan(self, query: Dict[str, Any], 
                             query_type: QueryType) -> List[Dict[str, Any]]:
        """Generate initial execution plan."""
        if query_type == QueryType.PATH_FINDING:
            return self._plan_path_finding(query)
        elif query_type == QueryType.PATTERN_MATCHING:
            return self._plan_pattern_matching(query)
        elif query_type == QueryType.AGGREGATION:
            return self._plan_aggregation(query)
        elif query_type == QueryType.SUBGRAPH:
            return self._plan_subgraph_query(query)
        else:
            return self._plan_entity_search(query)

    def _plan_path_finding(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan path finding query execution."""
        start_node = query['path']['start']
        end_node = query['path']['end']
        
        plan = [
            {
                'operation': 'locate_start',
                'node_id': start_node,
                'use_index': True
            },
            {
                'operation': 'locate_end',
                'node_id': end_node,
                'use_index': True
            },
            {
                'operation': 'find_paths',
                'max_length': query.get('max_length', 5),
                'use_cache': True
            }
        ]
        
        if 'constraints' in query:
            plan.append({
                'operation': 'apply_constraints',
                'constraints': query['constraints']
            })
            
        return plan

    def _plan_pattern_matching(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan pattern matching query execution."""
        pattern = query['pattern']
        
        plan = [
            {
                'operation': 'identify_pattern_nodes',
                'pattern': pattern,
                'use_index': True
            },
            {
                'operation': 'match_pattern',
                'use_cache': True
            }
        ]
        
        if 'filters' in query:
            plan.append({
                'operation': 'apply_filters',
                'filters': query['filters']
            })
            
        return plan

    def _plan_aggregation(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan aggregation query execution."""
        aggregations = query['aggregation']
        
        plan = [
            {
                'operation': 'collect_data',
                'fields': aggregations['fields'],
                'use_index': True
            },
            {
                'operation': 'perform_aggregation',
                'functions': aggregations['functions'],
                'use_cache': True
            }
        ]
        
        if 'grouping' in aggregations:
            plan.append({
                'operation': 'group_results',
                'grouping': aggregations['grouping']
            })
            
        return plan

    def _plan_subgraph_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan subgraph query execution."""
        nodes = query['subgraph']['nodes']
        
        plan = [
            {
                'operation': 'locate_nodes',
                'node_ids': nodes,
                'use_index': True
            },
            {
                'operation': 'expand_subgraph',
                'depth': query.get('depth', 1),
                'use_cache': True
            }
        ]
        
        if 'properties' in query:
            plan.append({
                'operation': 'fetch_properties',
                'properties': query['properties']
            })
            
        return plan

    def _plan_entity_search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan entity search query execution."""
        plan = [
            {
                'operation': 'search_entities',
                'criteria': query.get('criteria', {}),
                'use_index': True
            }
        ]
        
        if 'sort' in query:
            plan.append({
                'operation': 'sort_results',
                'sort_by': query['sort']
            })
            
        return plan

    def _apply_optimizations(self, plan: List[Dict[str, Any]], 
                           query_type: QueryType) -> List[Dict[str, Any]]:
        """Apply optimization strategies to the plan."""
        # Apply various optimization strategies
        plan = self._optimize_index_usage(plan)
        plan = self._optimize_cache_usage(plan)
        plan = self._optimize_step_order(plan)
        plan = self._optimize_data_access(plan)
        
        return plan

    def _optimize_index_usage(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize usage of indexes."""
        optimized_plan = []
        for step in plan:
            if step.get('use_index'):
                # Check index statistics
                if self._should_use_index(step):
                    step['index_type'] = self._select_best_index(step)
                else:
                    step['use_index'] = False
            optimized_plan.append(step)
        return optimized_plan

    def _optimize_cache_usage(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize usage of cache."""
        optimized_plan = []
        for step in plan:
            if step.get('use_cache'):
                # Check cache statistics
                if self._should_use_cache(step):
                    step['cache_strategy'] = self._select_cache_strategy(step)
                else:
                    step['use_cache'] = False
            optimized_plan.append(step)
        return optimized_plan

    def _optimize_step_order(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize order of execution steps."""
        # Sort steps by estimated cost and dependencies
        dependencies = self._analyze_step_dependencies(plan)
        costs = self._estimate_step_costs(plan)
        
        # Topological sort with cost consideration
        return self._topological_sort(plan, dependencies, costs)

    def _optimize_data_access(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize data access patterns."""
        optimized_plan = []
        for step in plan:
            # Add data prefetching where beneficial
            if self._should_prefetch(step):
                prefetch_step = {
                    'operation': 'prefetch_data',
                    'data': self._get_prefetch_data(step)
                }
                optimized_plan.append(prefetch_step)
            optimized_plan.append(step)
        return optimized_plan

    def _determine_cache_strategy(self, plan: List[Dict[str, Any]]) -> str:
        """Determine optimal cache strategy for the plan."""
        # Analyze plan characteristics
        data_size = self._estimate_data_size(plan)
        access_pattern = self._analyze_access_pattern(plan)
        reuse_potential = self._estimate_reuse_potential(plan)
        
        # Select cache strategy
        if data_size < self.optimizer_config.get('cache_size_threshold', 1000):
            if reuse_potential > 0.7:
                return 'full_cache'
            elif reuse_potential > 0.3:
                return 'partial_cache'
        
        if access_pattern == 'sequential':
            return 'stream_cache'
        
        return 'no_cache'

    def _identify_parallel_steps(self, plan: List[Dict[str, Any]]) -> List[int]:
        """Identify steps that can be executed in parallel."""
        parallel_steps = []
        dependencies = self._analyze_step_dependencies(plan)
        
        for i, step in enumerate(plan):
            # Check if step has no dependencies on previous unfinished steps
            if self._can_parallelize(step, i, dependencies):
                parallel_steps.append(i)
                
        return parallel_steps

    def _determine_index_usage(self, plan: List[Dict[str, Any]]) -> List[str]:
        """Determine which indexes will be used."""
        index_usage = []
        for step in plan:
            if step.get('use_index'):
                index_type = step.get('index_type')
                if index_type:
                    index_usage.append(index_type)
        return index_usage

    def _estimate_execution_cost(self, plan: List[Dict[str, Any]],
                               cache_strategy: str,
                               parallel_steps: List[int]) -> float:
        """Estimate total execution cost of the plan."""
        total_cost = 0.0
        step_costs = self._estimate_step_costs(plan)
        
        # Account for parallelization
        parallel_groups = self._group_parallel_steps(parallel_steps)
        
        for group in parallel_groups:
            group_cost = max(step_costs[i] for i in group)
            total_cost += group_cost
        
        # Apply cache factor
        cache_factor = self._get_cache_factor(cache_strategy)
        total_cost *= cache_factor
        
        return total_cost

    def _execute_step(self, step: Dict[str, Any], stats: Dict[str, Any]) -> Any:
        """Execute a single step in the plan."""
        operation = step['operation']
        
        # Update stats
        stats['nodes_accessed'] += self._count_nodes_accessed(step)
        if step.get('use_cache'):
            stats['cache_hits'] += 1
        if step.get('use_index'):
            stats['index_hits'] += 1
            
        # Execute appropriate operation
        if operation == 'locate_start':
            return self._execute_locate_node(step)
        elif operation == 'locate_end':
            return self._execute_locate_node(step)
        elif operation == 'find_paths':
            return self._execute_find_paths(step)
        elif operation == 'apply_constraints':
            return self._execute_apply_constraints(step)
        elif operation == 'identify_pattern_nodes':
            return self._execute_identify_pattern_nodes(step)
        elif operation == 'match_pattern':
            return self._execute_match_pattern(step)
        elif operation == 'apply_filters':
            return self._execute_apply_filters(step)
        elif operation == 'collect_data':
            return self._execute_collect_data(step)
        elif operation == 'perform_aggregation':
            return self._execute_aggregation(step)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _execute_locate_node(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node location step."""
        node_id = step['node_id']
        
        query = """
        MATCH (n:Entity {id: $node_id})
        RETURN n
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            return record['n'] if record else None

    def _execute_find_paths(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute path finding step."""
        max_length = step['max_length']
        start_node = step.get('start_node')
        end_node = step.get('end_node')
        
        query = """
        MATCH path = (start:Entity {id: $start_node})-[*..{max_length}]-(end:Entity {id: $end_node})
        RETURN path
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(
                query,
                start_node=start_node,
                end_node=end_node,
                max_length=max_length
            )
            return [record['path'] for record in result]

    def _execute_apply_constraints(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute constraint application step."""
        paths = step.get('paths', [])
        constraints = step['constraints']
        
        filtered_paths = []
        for path in paths:
            if self._check_constraints(path, constraints):
                filtered_paths.append(path)
                
        return filtered_paths

    def _execute_identify_pattern_nodes(self, step: Dict[str, Any]) -> List[str]:
        """Execute pattern node identification step."""
        pattern = step['pattern']
        
        # Convert pattern to Cypher pattern
        cypher_pattern = self._pattern_to_cypher(pattern)
        
        query = f"""
        MATCH {cypher_pattern}
        RETURN DISTINCT n.id as node_id
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            return [record['node_id'] for record in result]

    def _execute_match_pattern(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute pattern matching step."""
        pattern = step['pattern']
        nodes = step.get('nodes', [])
        
        # Build query with node constraints
        node_conditions = ' OR '.join(f"n.id = '{node}'" for node in nodes)
        
        query = f"""
        MATCH {pattern}
        WHERE {node_conditions}
        RETURN *
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def _execute_apply_filters(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute filter application step."""
        results = step.get('results', [])
        filters = step['filters']
        
        filtered_results = []
        for result in results:
            if self._check_filters(result, filters):
                filtered_results.append(result)
                
        return filtered_results

    def _execute_collect_data(self, step: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Execute data collection step."""
        fields = step['fields']
        
        # Build query for specified fields
        field_expressions = ', '.join(
            f"n.{field} as {field}"
            for field in fields
        )
        
        query = f"""
        MATCH (n:Entity)
        RETURN {field_expressions}
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            
            # Organize data by field
            data = {field: [] for field in fields}
            for record in result:
                for field in fields:
                    data[field].append(record[field])
                    
            return data

    def _execute_aggregation(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation step."""
        data = step.get('data', {})
        functions = step['functions']
        
        results = {}
        for field, funcs in functions.items():
            field_data = data.get(field, [])
            field_results = {}
            
            for func in funcs:
                if func == 'count':
                    field_results[func] = len(field_data)
                elif func == 'sum':
                    field_results[func] = sum(field_data)
                elif func == 'avg':
                    field_results[func] = sum(field_data) / len(field_data)
                elif func == 'min':
                    field_results[func] = min(field_data)
                elif func == 'max':
                    field_results[func] = max(field_data)
                    
            results[field] = field_results
            
        return results

    def _execute_parallel_step(self, step: Dict[str, Any], 
                             stats: Dict[str, Any]) -> Any:
        """Execute a step in parallel."""
        # Implement parallel execution logic based on step type
        # This is a placeholder for parallel execution logic
        return self._execute_step(step, stats)

    def _combine_results(self, results: List[Any], 
                        plan: QueryPlan) -> Any:
        """Combine results from multiple execution steps."""
        if not results:
            return None
            
        # Combine based on last operation type
        last_step = plan.steps[-1]
        operation = last_step['operation']
        
        if operation in ['find_paths', 'match_pattern']:
            return results[-1]  # Return final filtered results
        elif operation == 'perform_aggregation':
            return results[-1]  # Return aggregation results
        else:
            return results[-1]  # Default to last result

    def _init_query_stats(self):
        """Initialize query statistics tracking."""
        self.query_stats = {
            'total_queries': 0,
            'average_execution_time': 0.0,
            'cache_hit_rate': 0.0,
            'index_hit_rate': 0.0
        }

    def _update_query_history(self, plan: QueryPlan, stats: QueryStats):
        """Update query history and statistics."""
        self.query_stats['total_queries'] += 1
        
        # Update moving averages
        alpha = 0.1  # Smoothing factor
        self.query_stats['average_execution_time'] = (
            (1 - alpha) * self.query_stats['average_execution_time'] +
            alpha * stats.execution_time
        )
        
        # Store plan and stats for future optimization
        plan_key = self._get_plan_key(plan)
        self.query_history[plan_key].append({
            'stats': stats,
            'timestamp': datetime.now()
        })