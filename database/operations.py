# database/operations.py
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
import threading
from queue import Queue
import json
import traceback

from ..config import Config
from .graph import Neo4jConnection

class OperationType(Enum):
    """Types of graph operations."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    INDEX = "index"
    CONSTRAINT = "constraint"
    VALIDATION = "validation"

@dataclass
class OperationResult:
    """Container for operation results."""
    success: bool
    data: Optional[Any]
    error: Optional[str]
    execution_time: float
    retries: int
    operation_id: str

@dataclass
class OperationMetrics:
    """Container for operation metrics."""
    total_operations: int
    success_rate: float
    average_execution_time: float
    error_rate: float
    retry_rate: float
    concurrent_operations: int

class OperationMonitor:
    """Monitors and tracks graph operations."""
    
    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.metrics = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'total_execution_time': 0.0,
            'total_retries': 0
        }
        self._lock = threading.Lock()

    def start_operation(self, operation_id: str, operation_type: OperationType):
        """Record start of an operation."""
        with self._lock:
            self.operations[operation_id] = {
                'type': operation_type,
                'start_time': time.time(),
                'status': 'running',
                'retries': 0,
                'error': None
            }
            self.metrics['total'] += 1

    def end_operation(self, operation_id: str, success: bool, error: Optional[str] = None):
        """Record end of an operation."""
        with self._lock:
            if operation_id in self.operations:
                op = self.operations[operation_id]
                op['end_time'] = time.time()
                op['status'] = 'completed' if success else 'failed'
                op['error'] = error
                
                execution_time = op['end_time'] - op['start_time']
                self.metrics['total_execution_time'] += execution_time
                
                if success:
                    self.metrics['successful'] += 1
                else:
                    self.metrics['failed'] += 1

    def record_retry(self, operation_id: str):
        """Record an operation retry."""
        with self._lock:
            if operation_id in self.operations:
                self.operations[operation_id]['retries'] += 1
                self.metrics['total_retries'] += 1

    def get_metrics(self) -> OperationMetrics:
        """Get current operation metrics."""
        with self._lock:
            total = self.metrics['total']
            if total == 0:
                return OperationMetrics(
                    total_operations=0,
                    success_rate=0.0,
                    average_execution_time=0.0,
                    error_rate=0.0,
                    retry_rate=0.0,
                    concurrent_operations=0
                )
            
            return OperationMetrics(
                total_operations=total,
                success_rate=self.metrics['successful'] / total,
                average_execution_time=(
                    self.metrics['total_execution_time'] / total
                ),
                error_rate=self.metrics['failed'] / total,
                retry_rate=self.metrics['total_retries'] / total,
                concurrent_operations=len([
                    op for op in self.operations.values()
                    if op['status'] == 'running'
                ])
            )

class OperationHandler:
    """Handles execution and monitoring of graph operations."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        self.monitor = OperationMonitor()
        
        # Operation configurations
        self.max_retries = config.model_config.get('max_retries', 3)
        self.retry_delay = config.model_config.get('retry_delay', 1.0)
        self.validation_enabled = config.model_config.get('validation_enabled', True)
        
        # Operation queue for rate limiting
        self.operation_queue = Queue()
        self.rate_limit_thread = threading.Thread(
            target=self._process_operation_queue,
            daemon=True
        )
        self.rate_limit_thread.start()

    def execute_operation(self, operation_type: OperationType,
                        data: Dict[str, Any]) -> OperationResult:
        """Execute a graph operation with monitoring and retries."""
        operation_id = self._generate_operation_id()
        self.monitor.start_operation(operation_id, operation_type)
        
        retries = 0
        start_time = time.time()
        
        try:
            # Validate operation if enabled
            if self.validation_enabled:
                self._validate_operation(operation_type, data)
            
            while retries <= self.max_retries:
                try:
                    # Execute operation
                    result = self._execute_single_operation(
                        operation_type,
                        data
                    )
                    
                    execution_time = time.time() - start_time
                    self.monitor.end_operation(operation_id, True)
                    
                    return OperationResult(
                        success=True,
                        data=result,
                        error=None,
                        execution_time=execution_time,
                        retries=retries,
                        operation_id=operation_id
                    )
                    
                except Exception as e:
                    retries += 1
                    self.monitor.record_retry(operation_id)
                    
                    if retries <= self.max_retries:
                        time.sleep(self.retry_delay * retries)
                        continue
                    
                    raise e
                    
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Operation failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self.monitor.end_operation(operation_id, False, error_msg)
            
            return OperationResult(
                success=False,
                data=None,
                error=error_msg,
                execution_time=execution_time,
                retries=retries,
                operation_id=operation_id
            )

    def batch_execute_operations(self, operations: List[Tuple[OperationType, Dict[str, Any]]],
                               parallel: bool = False) -> List[OperationResult]:
        """Execute multiple operations in batch."""
        if parallel:
            return self._parallel_execute_operations(operations)
        else:
            return [
                self.execute_operation(op_type, data)
                for op_type, data in operations
            ]

    def _parallel_execute_operations(self, operations: List[Tuple[OperationType, Dict[str, Any]]]) -> List[OperationResult]:
        """Execute operations in parallel."""
        results = []
        threads = []
        results_lock = threading.Lock()
        
        def execute_and_store(op_type: OperationType, data: Dict[str, Any]):
            result = self.execute_operation(op_type, data)
            with results_lock:
                results.append(result)
        
        # Start threads for each operation
        for op_type, data in operations:
            thread = threading.Thread(
                target=execute_and_store,
                args=(op_type, data)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        return results

    def _execute_single_operation(self, operation_type: OperationType,
                                data: Dict[str, Any]) -> Any:
        """Execute a single graph operation."""
        operation_handlers = {
            OperationType.CREATE: self._handle_create,
            OperationType.READ: self._handle_read,
            OperationType.UPDATE: self._handle_update,
            OperationType.DELETE: self._handle_delete,
            OperationType.MERGE: self._handle_merge,
            OperationType.INDEX: self._handle_index,
            OperationType.CONSTRAINT: self._handle_constraint,
            OperationType.VALIDATION: self._handle_validation
        }
        
        handler = operation_handlers.get(operation_type)
        if not handler:
            raise ValueError(f"Unsupported operation type: {operation_type}")
            
        return handler(data)

    def _handle_create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle entity creation operation."""
        query = """
        CREATE (n:Entity)
        SET n = $properties
        RETURN n
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, properties=data)
            record = result.single()
            return dict(record['n'])

    def _handle_read(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle entity read operation."""
        query = """
        MATCH (n:Entity)
        WHERE n.id = $id
        RETURN n
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, id=data['id'])
            record = result.single()
            return dict(record['n']) if record else None

    def _handle_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle entity update operation."""
        query = """
        MATCH (n:Entity {id: $id})
        SET n += $properties
        RETURN n
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(
                query,
                id=data['id'],
                properties=data['properties']
            )
            record = result.single()
            return dict(record['n'])

    def _handle_delete(self, data: Dict[str, Any]) -> bool:
        """Handle entity deletion operation."""
        query = """
        MATCH (n:Entity {id: $id})
        DELETE n
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, id=data['id'])
            return result.consume().counters.nodes_deleted > 0

    def _handle_merge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle entity merge operation."""
        query = """
        MERGE (n:Entity {id: $id})
        ON CREATE SET n = $properties
        ON MATCH SET n += $properties
        RETURN n
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(
                query,
                id=data['id'],
                properties=data['properties']
            )
            record = result.single()
            return dict(record['n'])

    def _handle_index(self, data: Dict[str, Any]) -> bool:
        """Handle index creation operation."""
        query = f"""
        CREATE INDEX {data['name']} IF NOT EXISTS
        FOR (n:Entity)
        ON (n.{data['property']})
        """
        
        with self.neo4j._driver.session() as session:
            session.run(query)
            return True

    def _handle_constraint(self, data: Dict[str, Any]) -> bool:
        """Handle constraint creation operation."""
        query = f"""
        CREATE CONSTRAINT {data['name']} IF NOT EXISTS
        FOR (n:Entity)
        REQUIRE n.{data['property']} IS UNIQUE
        """
        
        with self.neo4j._driver.session() as session:
            session.run(query)
            return True

    def _handle_validation(self, data: Dict[str, Any]) -> bool:
        """Handle data validation operation."""
        validation_type = data['type']
        value = data['value']
        
        if validation_type == 'entity':
            return self._validate_entity(value)
        elif validation_type == 'relationship':
            return self._validate_relationship(value)
        else:
            raise ValueError(f"Unsupported validation type: {validation_type}")

    def _validate_operation(self, operation_type: OperationType, 
                          data: Dict[str, Any]):
        """Validate operation data."""
        required_fields = {
            OperationType.CREATE: {'properties'},
            OperationType.READ: {'id'},
            OperationType.UPDATE: {'id', 'properties'},
            OperationType.DELETE: {'id'},
            OperationType.MERGE: {'id', 'properties'},
            OperationType.INDEX: {'name', 'property'},
            OperationType.CONSTRAINT: {'name', 'property'},
            OperationType.VALIDATION: {'type', 'value'}
        }
        
        fields = required_fields.get(operation_type, set())
        missing_fields = fields - set(data.keys())
        
        if missing_fields:
            raise ValueError(
                f"Missing required fields for {operation_type}: {missing_fields}"
            )

    def _validate_entity(self, entity: Dict[str, Any]) -> bool:
        """Validate entity data."""
        required_fields = {'id', 'type'}
        return all(field in entity for field in required_fields)

    def _validate_relationship(self, relationship: Dict[str, Any]) -> bool:
        """Validate relationship data."""
        required_fields = {'source', 'target', 'type'}
        return all(field in relationship for field in required_fields)

    def _process_operation_queue(self):
        """Process operations from queue with rate limiting."""
        while True:
            operation = self.operation_queue.get()
            if operation is None:
                break
                
            try:
                operation_type, data = operation
                self.execute_operation(operation_type, data)
            except Exception as e:
                self.logger.error(f"Error processing queued operation: {str(e)}")
            finally:
                self.operation_queue.task_done()

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        timestamp = int(time.time() * 1000)
        return f"op_{timestamp}_{id(threading.current_thread())}"

    def get_operation_metrics(self) -> OperationMetrics:
        """Get current operation metrics."""
        return self.monitor.get_metrics()

    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific operation."""
        return self.monitor.operations.get(operation_id)