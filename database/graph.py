# database/graph.py
# database/graph.py
from neo4j import Transaction, GraphDatabase
from neo4j.exceptions import Neo4jError, TransientError
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import logging
from contextlib import contextmanager
import json
import uuid
import time
import re
from functools import wraps  # Add this import for the decorator
from neo4j import Session
import hashlib
import torch

from database.connection import Neo4jConnection
from config import Config
from llm.model import LlamaProcessor, ExtractionResult 
from .retrieval import GraphRetriever, RetrievalResult
from .staged_retrieval import StagedRetriever
from .summarization import GraphSummarizer
from .analytics import GraphAnalyzer
from .pathfinding import PathFinder
from .query_optimizer import QueryOptimizer
from .operations import OperationHandler, OperationType, OperationResult

from services.validation import GraphValidator, ValidationError
from utils.error_handler import handle_errors, error_tracker, GraphError
from services.cache import cache_manager, cache_result
from services.optimizer import PerformanceOptimizer, QueryOptimizationContext
from services.monitoring import MonitoringService

from document_processor.factory import DocumentProcessorFactory  # Ensure this import is present



# Add missing class definitions
class GraphProcessingError(GraphError):
    """Error raised during graph processing operations."""
    pass

class ExponentialBackoff:
    def __init__(self, initial_delay: float, max_delay: float, factor: float):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.factor ** attempt)
        return min(delay, self.max_delay)

class QueryCache:
    def __init__(self):
        self._cache = {}
        self._stats = {'hits': 0, 'misses': 0}

    def clear(self):
        self._cache.clear()
        
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()

    def cache_query_result(self, query: str, results: Any):
        self._cache[query] = results
            

@dataclass
class RetryPolicy:
    max_retries: int
    base_delay: float = 1.0
    max_delay: float = 60.0
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self.max_retries:
            return False
        return self._is_retryable(exception)
    
    def _is_retryable(self, exception: Exception) -> bool:
        return isinstance(exception, (
            TransientError,
            ConnectionError,
            TimeoutError
        ))
    
    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(
            self.base_delay * (2 ** attempt),
            self.max_delay
        )
        return delay
        
class KnowledgeGraph:
    """Manages knowledge graph operations."""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection = Neo4jConnection(config)
        self.optimizer = PerformanceOptimizer(config, self.connection)
        self.monitoring = MonitoringService(config, self.connection)
        
        # Initialize components
        self.validator = GraphValidator(config, self.connection)
        self.retriever = GraphRetriever(config, self.connection)
        self.query_cache = QueryCache()
        self.community_detector = CommunityDetector(config, self.connection)
        self.embedding_cache = cache_manager  # Use global cache manager for embeddings

        self._transaction_locks = {}  # Dict to store locks per document
        self._global_lock = threading.Lock()
        
        self._batch_size = 1000
        self._retry_policy = RetryPolicy(max_retries=3)

    @contextmanager
    def _get_doc_lock(self, doc_id: str):
        """Get or create document-specific lock."""
        with self._global_lock:
            if doc_id not in self._transaction_locks:
                self._transaction_locks[doc_id] = threading.Lock()
            lock = self._transaction_locks[doc_id]
        
        try:
            with lock:
                yield
        finally:
            # Cleanup unused locks
            with self._global_lock:
                if not lock.locked():
                    self._transaction_locks.pop(doc_id, None)

    def _batch_process_entities(self, tx, entities: List[Dict]) -> List[Dict]:
        """Process entities in optimized batches."""
        query = """
        UNWIND $entities as entity
        MERGE (e:Entity {id: entity.id})
        ON CREATE SET 
            e += entity.properties,
            e.created_at = datetime(),
            e.last_updated = datetime()
        ON MATCH SET 
            e += entity.properties,
            e.last_updated = datetime(),
            e.update_count = COALESCE(e.update_count, 0) + 1
        WITH e
        RETURN e, e.id as id
        """
        if not all(isinstance(e, dict) and 'id' in e for e in entities):
            raise ValueError("Invalid entity format")
        
        # Add batch size validation
        if len(entities) > self._batch_size:
            raise ValueError(f"Batch size {len(entities)} exceeds maximum {self._batch_size}")
        
        try:
            result = tx.run(query, entities=entities)
            processed = [record['e'] for record in result]
            return processed
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise GraphProcessingError("Entity batch processing failed") from e
    

    @handle_errors(logger=logging.getLogger(__name__))    
    def process_extraction_results(self, doc_id: str, results: List[ExtractionResult]):
        attempt = 0
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            max_delay=60.0,
            factor=2.0
        )
        
        while True:
            try:
                with self._get_doc_lock(doc_id):
                    # Process batch with transaction
                    return self._process_with_transaction(doc_id, results)
                    
            except TransientError as e:
                attempt += 1
                if attempt >= self._retry_policy.max_retries:
                    raise GraphProcessingError("Max retries exceeded") from e
                    
                delay = backoff.get_delay(attempt)
                self.logger.warning(
                    f"Retrying transaction (attempt {attempt}) after {delay}s: {str(e)}"
                )
                time.sleep(delay)
                continue
                
            except Exception as e:
                self.logger.error(f"Fatal error in transaction: {str(e)}")
                raise GraphProcessingError("Transaction failed") from e

    def _process_with_transaction(self, doc_id: str, results: List[ExtractionResult]) -> Dict[str, Any]:
        """Process with transaction timeout.
        
        Args:
            doc_id: Document identifier
            results: List of extraction results
            
        Returns:
            Dict containing processed entities and relationships
            
        Raises:
            GraphProcessingError: If processing fails
        """
        timeout = self.config.get('transaction_timeout', 30)  # 30 second default
        
        with self.connection._driver.session() as session:
            try:
                with session.begin_transaction(timeout=timeout) as tx:
                    # Process batches
                    processed = self._process_batches(tx, doc_id, results)
                    tx.commit()
                    return processed
            except Exception as e:
                self.logger.error(f"Transaction failed: {str(e)}")
                if 'tx' in locals():
                    tx.rollback()
                raise GraphProcessingError(f"Transaction failed: {str(e)}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of resources."""
        try:
            # Clean up caches
            self.query_cache.clear()
            self.embedding_cache.clear()
            
            # Close connections
            if hasattr(self, 'connection'):
                self.connection.close()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise

    def update_communities(self):
        """Update graph communities."""
        detector = CommunityDetector(self.config, self.connection)
        detector.update_community_summaries(self.llm_processor)

    def _validate_batch(self, batch: List[Dict[str, Any]], batch_type: str) -> List[Dict[str, Any]]:
        """Validate a batch of entities or relationships.
        
        Args:
            batch: List of items to validate
            batch_type: Type of items ('entity' or 'relationship')
            
        Returns:
            List of valid items
            
        Raises:
            ValueError: If batch_type is invalid
        """
        if batch_type not in ['entity', 'relationship']:
            raise ValueError(f"Invalid batch type: {batch_type}")
            
        valid_items = []
        validation_func = (self.validator.validate_entity 
                          if batch_type == 'entity' 
                          else self.validator.validate_relationship)
        
        for item in batch:
            try:
                validation_result = validation_func(item)
                if validation_result.success:
                    valid_items.append(item)
                else:
                    self.logger.warning(
                        f"{batch_type.capitalize()} validation failed: {validation_result.message}",
                        extra={'item': item}
                    )
            except Exception as e:
                self.logger.error(
                    f"Error validating {batch_type}: {str(e)}",
                    extra={'item': item}
                )
                
        return valid_items

    # Need to add this method
    def _process_batches(self, tx: Transaction, doc_id: str, results: List[ExtractionResult]) -> Dict[str, Any]:
        """Process results in batches.
        
        Args:
            tx: Neo4j transaction
            doc_id: Document identifier
            results: List of extraction results
            
        Returns:
            Dict containing processed entities and relationships
            
        Raises:
            GraphProcessingError: If batch processing fails
        """
        try:
            # Create document node
            self._create_document_node(tx, doc_id)
            
            processed_entities = []
            processed_relationships = []
            metrics = {
                'entity_count': 0,
                'relationship_count': 0,
                'batch_count': 0
            }
            
            # Process in batches
            for result in results:
                # Validate entities
                valid_entities = []
                for entity in result.entities:
                    validation_result = self.validator.validate_entity(entity)
                    if validation_result.success:
                        valid_entities.append(entity)
                
                # Process entities in batches
                for i in range(0, len(valid_entities), self._batch_size):
                    batch = valid_entities[i:i + self._batch_size]
                    entity_batch = self._batch_process_entities(tx, batch)
                    processed_entities.extend(entity_batch)
                    
                    # Link entities to document
                    self._link_entities_to_document(
                        tx, 
                        doc_id,
                        [e['id'] for e in entity_batch],
                        result.chunk_id
                    )
                    
                    metrics['entity_count'] += len(entity_batch)
                    metrics['batch_count'] += 1
                
                # Process relationships
                valid_relationships = []
                for rel in result.relationships:
                    validation_result = self.validator.validate_relationship(rel)
                    if validation_result.success:
                        valid_relationships.append(rel)
                
                # Create relationships
                for relationship in valid_relationships:
                    processed_rel = self._create_relationships(
                        tx,
                        [relationship],
                        result.confidence
                    )
                    processed_relationships.extend(processed_rel)
                    metrics['relationship_count'] += 1
            
            return {
                'doc_id': doc_id,
                'entities': processed_entities,
                'relationships': processed_relationships,
                'metrics': metrics,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise GraphProcessingError(f"Failed to process batches: {str(e)}") from e

    def _create_document_node(self, session: Session, doc_id: str):
        """Create a document node in the graph."""
        query = """
        MERGE (d:Document {id: $doc_id})
        ON CREATE SET 
            d.created_at = datetime(),
            d.last_updated = datetime()
        ON MATCH SET 
            d.last_updated = datetime()
        """
        session.run(query, doc_id=doc_id)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        cache_metrics = cache_manager.get_stats()
        optimization_metrics = self.optimizer.get_optimization_metrics()
        
        return {
            'cache': cache_metrics,
            'optimization': optimization_metrics
        }

    def _process_single_result(self, session: Session, doc_id: str, result: ExtractionResult):
        """Process a single extraction result."""
        # Process entities
        entity_ids = self._create_entities(session, result.entities)
        
        # Link entities to document
        self._link_entities_to_document(session, doc_id, entity_ids, result.chunk_id)
        
        # Process relationships
        self._create_relationships(session, result.relationships, result.confidence)

    def staged_query_graph(self, query: str, embedding_model) -> Dict[str, Any]:
        """Query graph using staged retrieval process."""
        retriever = StagedRetriever(self.config, self.connection)
        results = retriever.staged_retrieve(query, embedding_model)
        
        return {
            'subgraphs': results.final_results.subgraphs,
            'communities': [c.id for c in results.final_results.communities],
            'confidence': results.overall_confidence,
            'retrieval_time': results.total_time,
            'stage_metrics': [
                {
                    'stage': m.stage.value,
                    'duration': m.duration,
                    'nodes_processed': m.nodes_processed,
                    'success_rate': m.success_rate
                }
                for m in results.stage_metrics
            ]
        }

    def _create_entities(self, session: Session, entities: List[Dict[str, Any]]) -> List[str]:
        """Create entity nodes in the graph."""
        entity_ids = []
        
        for entity in entities:
            entity_id = str(uuid.uuid4())
            query = """
            MERGE (e:Entity {text: $text, type: $type})
            ON CREATE SET 
                e.id = $id,
                e.created_at = datetime(),
                e.last_updated = datetime(),
                e.occurrence_count = 1
            ON MATCH SET 
                e.last_updated = datetime(),
                e.occurrence_count = e.occurrence_count + 1
            RETURN e.id
            """
            
            result = session.run(
                query,
                text=entity['text'],
                type=entity['type'],
                id=entity_id
            )
            
            entity_ids.append(result.single()[0])
            
        return entity_ids

    def _link_entities_to_document(self, session: Session, doc_id: str, 
                                 entity_ids: List[str], chunk_id: str):
        """Create relationships between entities and document."""
        query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        MERGE (e)-[r:MENTIONED_IN]->(d)
        ON CREATE SET 
            r.created_at = datetime(),
            r.chunks = [$chunk_id],
            r.mention_count = 1
        ON MATCH SET 
            r.chunks = CASE 
                WHEN NOT $chunk_id IN r.chunks 
                THEN r.chunks + $chunk_id 
                ELSE r.chunks 
                END,
            r.mention_count = r.mention_count + 1
        """
        
        session.run(
            query,
            doc_id=doc_id,
            entity_ids=entity_ids,
            chunk_id=chunk_id
        )

    def _create_relationships(self, session: Session, relationships: List[Dict[str, Any]], 
                            confidence: float):
        """Create relationships between entities."""
        for rel in relationships:
            query = """
            MATCH (e1:Entity {text: $source})
            MATCH (e2:Entity {text: $target})
            MERGE (e1)-[r:RELATED_TO {type: $rel_type}]->(e2)
            ON CREATE SET 
                r.created_at = datetime(),
                r.confidence = $confidence,
                r.occurrence_count = 1
            ON MATCH SET 
                r.confidence = CASE
                    WHEN r.confidence < $confidence 
                    THEN $confidence 
                    ELSE r.confidence 
                    END,
                r.occurrence_count = r.occurrence_count + 1
            """
            
            session.run(
                query,
                source=rel['source'],
                target=rel['target'],
                rel_type=rel['relationship'],
                confidence=confidence
            )

    @cache_result(ttl=3600)
    def query_entities(self, entity_type: Optional[str] = None,
                      text_contains: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query entities with caching."""
        try:
            query = "MATCH (e:Entity)"
            params = {}
            
            with QueryOptimizationContext(self.optimizer, query, params) as (opt_query, opt_params):
                with self.connection._driver.session() as session:
                    result = session.run(opt_query, opt_params)
                    return [dict(record['e']) for record in result]
                    
        except Exception as e:
            self.logger.error(f"Error querying entities: {str(e)}")
            raise
        
    def query_relationships(self, source_text: Optional[str] = None,
                          relationship_type: Optional[str] = None,
                          min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Query relationships with optional filters."""
        with self.connection._driver.session() as session:
            query = """
            MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
            WHERE r.confidence >= $min_confidence
            """
            
            params = {'min_confidence': min_confidence}
            
            if source_text:
                query += " AND e1.text = $source"
                params['source'] = source_text
                
            if relationship_type:
                query += " AND r.type = $rel_type"
                params['rel_type'] = relationship_type
                
            query += " RETURN e1.text as source, r.type as relationship, e2.text as target, r.confidence as confidence"
            
            result = session.run(query, params)
            return [dict(record) for record in result]

    def get_entity_network(self, entity_text: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get network of entities connected to a given entity."""
        with self.connection._driver.session() as session:
            query = """
            MATCH path = (start:Entity {text: $text})-[*1..$depth]-(connected:Entity)
            RETURN path
            """
            
            result = session.run(query, text=entity_text, depth=max_depth)
            
            # Process results into a network format
            nodes = set()
            edges = set()
            
            for record in result:
                path = record['path']
                
                for node in path.nodes:
                    nodes.add((node['id'], node['text'], node['type']))
                    
                for rel in path.relationships:
                    edges.add((rel.start_node['id'], rel.end_node['id'], rel.type))
            
            return {
                'nodes': [{'id': n[0], 'text': n[1], 'type': n[2]} for n in nodes],
                'edges': [{'source': e[0], 'target': e[1], 'type': e[2]} for e in edges]
            }

    def generate_summaries(self, llm_processor) -> Dict[str, Any]:
        """Generate graph summaries at all levels."""
        summarizer = GraphSummarizer(self.config, self.connection)
        communities = self.community_detector.detect_communities()
        summaries = summarizer.generate_summaries(communities, llm_processor)
        
        return {
            'node_summaries': len([s for s in summaries.values() if s.level == 'node']),
            'community_summaries': len([s for s in summaries.values() if s.level == 'community']),
            'global_summary': summaries['global'].content if 'global' in summaries else None
        }
    
    def get_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific summary."""
        summarizer = GraphSummarizer(self.config, self.connection)
        summary = summarizer.get_summary(summary_id)
        
        if summary:
            return {
                'level': summary.level,
                'content': summary.content,
                'metadata': summary.metadata,
                'created_at': summary.created_at.isoformat(),
                'last_updated': summary.last_updated.isoformat()
            }
        return None

    def analyze_graph(self) -> Dict[str, Any]:
        """Perform comprehensive graph analysis."""
        analyzer = GraphAnalyzer(self.config, self.connection)
        
        # Get graph-level metrics
        graph_metrics = analyzer.compute_graph_metrics()
        
        # Get top important nodes
        important_nodes = analyzer.get_node_importance_ranking()[:10]
        
        # Get community metrics
        community_metrics = analyzer.get_community_metrics()
        
        # Get important patterns
        patterns = analyzer.find_important_patterns()
        
        return {
            'graph_metrics': {
                'total_nodes': graph_metrics.total_nodes,
                'total_edges': graph_metrics.total_edges,
                'density': graph_metrics.density,
                'modularity': graph_metrics.modularity,
                'avg_path_length': graph_metrics.average_path_length,
                'diameter': graph_metrics.diameter,
                'num_components': graph_metrics.num_connected_components,
                'assortativity': graph_metrics.assortativity,
                'node_types': graph_metrics.node_type_distribution,
                'relationship_types': graph_metrics.relationship_type_distribution
            },
            'important_nodes': [
                {
                    'node_id': node_id,
                    'importance_score': score
                }
                for node_id, score in important_nodes
            ],
            'community_analysis': {
                str(comm_id): metrics 
                for comm_id, metrics in community_metrics.items()
            },
            'patterns': {
                'hubs': patterns['hubs'],
                'bridges': patterns['bridges'],
                'dense_subgraphs': [list(sg) for sg in patterns['dense_subgraphs']],
                'motifs': patterns['motifs']
            }
        }
    
    def get_node_metrics(self, node_id: str) -> Dict[str, Any]:
            """Get detailed metrics for a specific node."""
            analyzer = GraphAnalyzer(self.config, self.connection)
            
            # Get metrics for the specific node
            node_metrics = analyzer.compute_node_metrics([node_id])
            if node_id not in node_metrics:
                return None
                
            metrics = node_metrics[node_id]
            
            return {
                'centrality': {
                    'degree': metrics.degree_centrality,
                    'betweenness': metrics.betweenness_centrality,
                    'eigenvector': metrics.eigenvector_centrality,
                    'pagerank': metrics.pagerank
                },
                'structural': {
                    'clustering_coefficient': metrics.clustering_coefficient,
                    'core_number': metrics.core_number,
                    'community_id': metrics.community_id
                }
            }
        
    def get_community_analysis(self, community_id: int) -> Dict[str, Any]:
            """Get detailed analysis for a specific community."""
            analyzer = GraphAnalyzer(self.config, self.connection)
            
            # Get community metrics
            community_metrics = analyzer.get_community_metrics()
            
            if community_id not in community_metrics:
                return None
                
            metrics = community_metrics[community_id]
            
            return {
                'size': metrics['size'],
                'density': metrics['density'],
                'avg_clustering': metrics['avg_clustering'],
                'diameter': metrics['diameter'],
                'hub_nodes': metrics['hub_nodes']
            }
    
    def find_paths(self, start_node: str, end_node: str, 
                   query_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Find paths between two nodes."""
        pathfinder = PathFinder(self.config, self.connection)
        paths = pathfinder.find_paths(start_node, end_node, query_embedding)
        
        return {
            'paths': [
                {
                    'nodes': path.nodes,
                    'relationships': path.relationships,
                    'cost': {
                        'distance': path.cost.distance,
                        'semantic_cost': path.cost.semantic_cost,
                        'confidence': path.cost.confidence,
                        'total_cost': path.cost.total_cost
                    },
                    'metadata': path.metadata
                }
                for path in paths
            ]
        }
    
    def find_multi_target_paths(self, start_node: str, target_nodes: List[str],
                              query_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Find paths from start node to multiple targets."""
        pathfinder = PathFinder(self.config, self.connection)
        paths = pathfinder.find_multi_target_paths(start_node, target_nodes, query_embedding)
        
        return {
            'paths': [
                {
                    'nodes': path.nodes,
                    'relationships': path.relationships,
                    'cost': {
                        'distance': path.cost.distance,
                        'semantic_cost': path.cost.semantic_cost,
                        'confidence': path.cost.confidence,
                        'total_cost': path.cost.total_cost
                    },
                    'metadata': path.metadata
                }
                for path in paths
            ]
        }

    def execute_optimized_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an optimized graph query."""
        optimizer = QueryOptimizer(self.config, self.connection)
        
        # Generate optimized plan
        plan = optimizer.optimize_query(query)
        
        # Execute plan and get results
        results, stats = optimizer.execute_plan(plan)
        
        return {
            'results': results,
            'execution_stats': {
                'execution_time': stats.execution_time,
                'nodes_accessed': stats.nodes_accessed,
                'cache_hits': stats.cache_hits,
                'index_hits': stats.index_hits,
                'memory_usage': stats.memory_usage
            },
            'plan': {
                'estimated_cost': plan.estimated_cost,
                'cache_strategy': plan.cache_strategy,
                'parallel_steps': plan.parallel_steps,
                'index_usage': plan.index_usage
            }
        }

    def execute_operation(self, operation_type: OperationType, 
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a monitored graph operation."""
        handler = OperationHandler(self.config, self.connection)
        result = handler.execute_operation(operation_type, data)
        
        return {
            'success': result.success,
            'data': result.data,
            'execution_time': result.execution_time,
            'operation_id': result.operation_id,
            'metrics': {
                'retries': result.retries,
                'error': result.error
            }
        }
    
    def batch_execute_operations(self, operations: List[Tuple[OperationType, Dict[str, Any]]],
                               parallel: bool = False) -> Dict[str, Any]:
        """Execute multiple operations in batch."""
        handler = OperationHandler(self.config, self.connection)
        results = handler.batch_execute_operations(operations, parallel)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return {
            'total_operations': len(results),
            'successful_operations': len(successful),
            'failed_operations': len(failed),
            'average_execution_time': sum(r.execution_time for r in results) / len(results),
            'operations': [
                {
                    'operation_id': r.operation_id,
                    'success': r.success,
                    'data': r.data,
                    'execution_time': r.execution_time,
                    'retries': r.retries,
                    'error': r.error
                }
                for r in results
            ]
        }
    
    def get_operation_metrics(self) -> Dict[str, Any]:
        """Get current operation metrics."""
        handler = OperationHandler(self.config, self.connection)
        metrics = handler.get_operation_metrics()
        
        return {
            'total_operations': metrics.total_operations,
            'success_rate': metrics.success_rate,
            'average_execution_time': metrics.average_execution_time,
            'error_rate': metrics.error_rate,
            'retry_rate': metrics.retry_rate,
            'concurrent_operations': metrics.concurrent_operations
        }
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific operation."""
        handler = OperationHandler(self.config, self.connection)
        return handler.get_operation_status(operation_id)
    
    def query_graph(self, query: str, embedding_model) -> Dict[str, Any]:
        """
        Execute graph query with both retrieval and performance monitoring.
        
        Args:
            query: Query string to execute
            embedding_model: Model for generating embeddings
            
        Returns:
            Dict containing:
            - results: Query results
            - metadata: Performance and execution metadata
            - retrieval_info: Information about retrieval process
            
        Raises:
            GraphError: If query execution fails
        """
        start_time = time.time()
        metrics = {
            'embedding_time': 0.0,
            'retrieval_time': 0.0,
            'processing_time': 0.0,
            'total_time': 0.0
        }
    
        try:
            # 1. Generate query embedding
            embed_start = time.time()
            query_embedding = self.retriever._get_query_embedding(query, embedding_model)
            metrics['embedding_time'] = time.time() - embed_start
    
            # 2. Check cache
            cache_key = self._generate_cache_key(query, query_embedding)
            cached_result = self.query_cache.get(cache_key) if self.config.get('use_cache', True) else None
            if cached_result:
                return {
                    'results': cached_result,
                    'metadata': {'cache_hit': True},
                    'metrics': metrics
                }
    
            # 3. Query Optimization
            optimization_start = time.time()
            optimized_query = self.optimizer.optimize_query(
                query, 
                {'embedding': query_embedding.tolist()}
            )
    
            # 4. Execute Retrieval
            retrieval_start = time.time()
            retrieval_results = self.retriever.retrieve(
                query,
                query_embedding,
                max_results=self.config.model_config.get('max_results', 10)
            )
            metrics['retrieval_time'] = time.time() - retrieval_start
    
            # 5. Process Results
            processing_start = time.time()
            processed_results = self._process_query_results(
                retrieval_results,
                query_embedding
            )
            metrics['processing_time'] = time.time() - processing_start
    
            # 6. Calculate final metrics
            metrics['total_time'] = time.time() - start_time
    
            # 7. Monitor Performance
            self.monitoring.performance_monitor.record_request(
                duration=metrics['total_time'],
                success=True,
                endpoint='query_graph',
                metrics=metrics
            )
    
            # 8. Cache results if enabled
            if self.config.model_config.get('cache_enabled', True):
                self.query_cache.cache_query_result(
                    cache_key,
                    processed_results
                )
    
            # 9. Prepare response
            response = {
                'results': processed_results,
                'metadata': {
                    'duration': metrics['total_time'],
                    'metrics': metrics,
                    'optimized_query': optimized_query
                },
                'retrieval_info': {
                    'subgraphs': [sg['id'] for sg in retrieval_results.subgraphs],
                    'communities': [c.id for c in retrieval_results.communities],
                    'confidence': retrieval_results.confidence
                }
            }
    
            return response
    
        except Exception as e:
            # Record failure metrics
            duration = time.time() - start_time
            metrics['total_time'] = duration
            
            self.monitoring.performance_monitor.record_request(
                duration=duration,
                success=False,
                endpoint='query_graph',
                error=str(e),
                metrics=metrics
            )
    
            # Track error
            error_details = {
                'query': query,
                'duration': duration,
                'metrics': metrics
            }
            
            error_tracker.track_error(GraphError(
                str(e),
                'query_execution_error',
                error_details
            ))
    
            raise GraphError(
                f"Query execution failed: {str(e)}",
                'query_execution_error',
                error_details
            ) from e
    
    def _generate_cache_key(self, query: str, embedding: torch.Tensor) -> str:
        """Generate unique cache key for query and embedding."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        embedding_hash = hashlib.md5(
            embedding.cpu().numpy().tobytes()
        ).hexdigest()
        return f"query_{query_hash}_{embedding_hash}"
        
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring metrics.
        Returns:
            Dict containing metrics from all monitoring components
        """
        try:
            # Get basic metrics from monitors
            metrics = {
                'system': self.monitoring.system_monitor.collect_metrics(),
                'database': self.monitoring.database_monitor.collect_metrics(),
                'operations': self.monitoring.operation_monitor.metrics,
                'performance': self.monitoring.performance_monitor.metrics
            }
            
            # Add graph-specific metrics
            graph_metrics = self._collect_graph_metrics()
            metrics['graph'] = graph_metrics
            
            # Add cache metrics
            cache_metrics = {
                'query_cache': self.query_cache.get_stats(),
                'embedding_cache': self.embedding_cache.get_stats() if hasattr(self, 'embedding_cache') else {}
            }
            metrics['cache'] = cache_metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting monitoring metrics: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        Returns:
            Dict containing health status and any warnings/errors
        """
        try:
            # Get basic health status
            status = self.monitoring.get_health_status()
            
            # Add graph-specific health checks
            graph_health = self._check_graph_health()
            status['graph_health'] = graph_health
            
            # Check database connectivity
            db_health = self._check_database_health()
            status['database_health'] = db_health
            
            # Update overall health status
            status['healthy'] = all([
                status['healthy'],
                graph_health['healthy'],
                db_health['healthy']
            ])
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error checking health status: {str(e)}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _monitor_transaction(self, func):
        """
        Decorator to monitor database transactions.
        Args:
            func: Function to monitor
        Returns:
            Wrapped function with monitoring
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            transaction_metrics = {
                'operation': func.__name__,
                'args_length': len(args),
                'kwargs_length': len(kwargs)
            }
            
            try:
                # Begin transaction monitoring
                self.monitoring.database_monitor.begin_transaction(
                    operation=func.__name__
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration = time.time() - start_time
                transaction_metrics['duration'] = duration
                
                # Record successful transaction
                self.monitoring.database_monitor.record_transaction(
                    duration=duration,
                    success=True,
                    operation=func.__name__,
                    metrics=transaction_metrics
                )
                
                return result
                
            except Exception as e:
                # Calculate duration
                duration = time.time() - start_time
                transaction_metrics['duration'] = duration
                transaction_metrics['error'] = str(e)
                
                # Record failed transaction
                self.monitoring.database_monitor.record_transaction(
                    duration=duration,
                    success=False,
                    operation=func.__name__,
                    metrics=transaction_metrics,
                    error=str(e)
                )
                
                # Track error
                error_tracker.track_error(GraphError(
                    str(e),
                    'transaction_error',
                    {
                        'operation': func.__name__,
                        'duration': duration,
                        'metrics': transaction_metrics
                    }
                ))
                
                raise
                
            finally:
                # End transaction monitoring
                self.monitoring.database_monitor.end_transaction(
                    operation=func.__name__
                )
                
        return wrapper
    
    def _collect_graph_metrics(self) -> Dict[str, Any]:
        """Collect graph-specific metrics."""
        try:
            with self.connection._driver.session() as session:
                # Get node count
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()['count']
                
                # Get relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                relationship_count = result.single()['count']
                
                # Get type distributions
                result = session.run("""
                    MATCH (n) 
                    WITH labels(n) as types, count(*) as count
                    RETURN types, count
                """)
                type_distribution = {
                    '-'.join(record['types']): record['count']
                    for record in result
                }
                
                return {
                    'node_count': node_count,
                    'relationship_count': relationship_count,
                    'type_distribution': type_distribution,
                    'density': relationship_count / (node_count ** 2) if node_count > 0 else 0
                }
                
        except Exception as e:
            self.logger.error(f"Error collecting graph metrics: {str(e)}")
            return {}
    
    def _check_graph_health(self) -> Dict[str, Any]:
        """Check graph health status."""
        try:
            with self.connection._driver.session() as session:
                health_checks = {
                    'connectivity': self._check_graph_connectivity(session),
                    'integrity': self._check_data_integrity(session),
                    'indexes': self._check_index_health(session)
                }
                
                return {
                    'healthy': all(check['healthy'] for check in health_checks.values()),
                    'checks': health_checks,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error checking graph health: {str(e)}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            with self.connection._driver.session() as session:
                # Test basic query
                result = session.run("RETURN 1 as test")
                test_value = result.single()['test']
                
                return {
                    'healthy': test_value == 1,
                    'connection': True,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error checking database health: {str(e)}")
            return {
                'healthy': False,
                'connection': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # Helper methods
    def _check_graph_connectivity(self, session: Session) -> Dict[str, Any]:
        """Check graph connectivity health."""
        try:
            result = session.run("""
                MATCH (n)
                WITH count(n) as total_nodes
                MATCH (n)-[r]-(m)
                WITH total_nodes, count(DISTINCT n) + count(DISTINCT m) as connected_nodes
                RETURN toFloat(connected_nodes) / total_nodes as connectivity
            """)
            connectivity = result.single()['connectivity']
            
            return {
                'healthy': connectivity >= self.config.model_config.get('min_connectivity', 0.8),
                'connectivity': connectivity
            }
        except Exception as e:
            self.logger.error(f"Error checking graph connectivity: {str(e)}")
            return {'healthy': False, 'error': str(e)}
    
    def _check_data_integrity(self, session: Session) -> Dict[str, Any]:
        """Check data integrity."""
        try:
            # Check for broken relationships
            result = session.run("""
                MATCH ()-[r]->()
                WHERE NOT exists(r.created_at) OR NOT exists(r.confidence)
                RETURN count(r) as invalid_relationships
            """)
            invalid_rels = result.single()['invalid_relationships']
            
            # Check for invalid entities
            result = session.run("""
                MATCH (e:Entity)
                WHERE NOT exists(e.id) OR NOT exists(e.type)
                RETURN count(e) as invalid_entities
            """)
            invalid_entities = result.single()['invalid_entities']
            
            return {
                'healthy': invalid_rels == 0 and invalid_entities == 0,
                'invalid_relationships': invalid_rels,
                'invalid_entities': invalid_entities
            }
        except Exception as e:
            self.logger.error(f"Error checking data integrity: {str(e)}")
            return {'healthy': False, 'error': str(e)}
    
    def _check_index_health(self, session: Session) -> Dict[str, Any]:
        """Check index health."""
        try:
            result = session.run("CALL db.indexes()")
            indexes = list(result)
            
            # Check for failed indexes
            failed_indexes = [idx for idx in indexes if idx['state'] != 'ONLINE']
            
            return {
                'healthy': len(failed_indexes) == 0,
                'total_indexes': len(indexes),
                'failed_indexes': len(failed_indexes)
            }
        except Exception as e:
            self.logger.error(f"Error checking index health: {str(e)}")
            return {'healthy': False, 'error': str(e)}
    
    def _validate_doc_id(self, doc_id: str) -> bool:
        """
        Validate document ID format and existence.
        Args:
            doc_id: Document identifier to validate
        Returns:
            bool: True if valid, False otherwise
        """
        if not doc_id or not isinstance(doc_id, str):
            return False
            
        # Check format
        if not re.match(r'^[a-zA-Z0-9_-]+$', doc_id):
            return False
            
        # Check if already exists
        try:
            with self.connection._driver.session() as session:
                result = session.run(
                    "MATCH (d:Document {id: $doc_id}) RETURN count(d) as count",
                    doc_id=doc_id
                )
                count = result.single()['count']
                return count <= 1  # Allow 0 (new) or 1 (existing)
        except Exception as e:
            self.logger.error(f"Error validating document ID: {str(e)}")
            return False
    
    def _process_query_results(self, results: RetrievalResult,
                             query_embedding: torch.Tensor) -> List[Dict[str, Any]]:
        """Process query results with embeddings."""
        processed_results = []
        
        for subgraph in results.subgraphs:
            # Get subgraph embedding
            subgraph_embedding = self._get_subgraph_embedding(subgraph)
            
            # Calculate relevance score
            relevance = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                subgraph_embedding.unsqueeze(0)
            ).item()
            
            processed_results.append({
                'subgraph': subgraph,
                'relevance_score': relevance,
                'confidence': results.confidence
            })
        
        # Sort by relevance
        processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return processed_results
    
    def _get_subgraph_embedding(self, subgraph: Dict[str, Any]) -> torch.Tensor:
        """Get embedding for a subgraph."""
        # First check cache
        cache_key = f"subgraph_{hash(json.dumps(subgraph, sort_keys=True))}"
        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Calculate embedding from node texts
        node_texts = []
        for node in subgraph['nodes']:
            node_texts.append(node.get('text', ''))
        
        text = ' '.join(node_texts)
        embedding = self.retriever._get_text_embedding(text)
        
        # Cache result
        self.embedding_cache.put(cache_key, embedding)
        
        return embedding