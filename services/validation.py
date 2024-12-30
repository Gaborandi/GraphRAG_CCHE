# services/validation.py
from typing import List, Dict, Any, Optional, Union, Type
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from enum import Enum
import re
import traceback

#from ..config import Config
from config import Config                

# Instead of from .database.graph import Neo4jConnection
from database.graph import Neo4jConnection

class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCode(Enum):
    """Error classification codes."""
    INVALID_INPUT = "invalid_input"
    DATA_INTEGRITY = "data_integrity"
    GRAPH_STRUCTURE = "graph_structure"
    RELATIONSHIP_ERROR = "relationship_error"
    DUPLICATE_ENTRY = "duplicate_entry"
    MISSING_REQUIRED = "missing_required"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TYPE_MISMATCH = "type_mismatch"
    FORMAT_ERROR = "format_error"
    VALUE_ERROR = "value_error"

@dataclass
class ValidationRule:
    """Definition of a validation rule."""
    name: str
    description: str
    level: ValidationLevel
    error_code: ErrorCode
    validation_fn: callable
    params: Dict[str, Any]

@dataclass
class ValidationResult:
    """Result of validation check."""
    success: bool
    level: ValidationLevel
    error_code: Optional[ErrorCode]
    message: str
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class ValidationError(Exception):
    """Custom exception for validation errors."""
    error_code: ErrorCode
    message: str
    details: Dict[str, Any]
    level: ValidationLevel

class GraphValidator:
    """Handles validation of graph data and operations."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize validation rules
        self.rules: Dict[str, ValidationRule] = {}
        self._init_validation_rules()
        
        # Cache for validation results
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        # Load validation configurations
        self.validation_config = config.model_config.get('validation', {})

    def validate_entity(self, entity: Dict[str, Any]) -> ValidationResult:
        """Validate entity data."""
        try:
            # Check cache first
            cache_key = self._get_cache_key('entity', entity)
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
            
            # Apply entity validation rules
            for rule in self._get_rules_by_type('entity'):
                result = self._apply_rule(rule, entity)
                if not result.success:
                    self.validation_cache[cache_key] = result
                    return result
            
            # All rules passed
            result = ValidationResult(
                success=True,
                level=ValidationLevel.INFO,
                error_code=None,
                message="Entity validation successful",
                details={'entity_id': entity.get('id')},
                timestamp=datetime.now()
            )
            
            self.validation_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating entity: {str(e)}")
            return ValidationResult(
                success=False,
                level=ValidationLevel.ERROR,
                error_code=ErrorCode.INVALID_INPUT,
                message=f"Validation error: {str(e)}",
                details={'error': traceback.format_exc()},
                timestamp=datetime.now()
            )

    def validate_relationship(self, relationship: Dict[str, Any]) -> ValidationResult:
        """Validate relationship data."""
        try:
            # Check cache first
            cache_key = self._get_cache_key('relationship', relationship)
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
            
            # Apply relationship validation rules
            for rule in self._get_rules_by_type('relationship'):
                result = self._apply_rule(rule, relationship)
                if not result.success:
                    self.validation_cache[cache_key] = result
                    return result
            
            # All rules passed
            result = ValidationResult(
                success=True,
                level=ValidationLevel.INFO,
                error_code=None,
                message="Relationship validation successful",
                details={
                    'source': relationship.get('source'),
                    'target': relationship.get('target')
                },
                timestamp=datetime.now()
            )
            
            self.validation_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating relationship: {str(e)}")
            return ValidationResult(
                success=False,
                level=ValidationLevel.ERROR,
                error_code=ErrorCode.INVALID_INPUT,
                message=f"Validation error: {str(e)}",
                details={'error': traceback.format_exc()},
                timestamp=datetime.now()
            )

    def validate_graph_structure(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate overall graph structure."""
        try:
            # Check cache first
            cache_key = self._get_cache_key('structure', data)
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
            
            # Apply structure validation rules
            for rule in self._get_rules_by_type('structure'):
                result = self._apply_rule(rule, data)
                if not result.success:
                    self.validation_cache[cache_key] = result
                    return result
            
            # All rules passed
            result = ValidationResult(
                success=True,
                level=ValidationLevel.INFO,
                error_code=None,
                message="Graph structure validation successful",
                details={'validated_at': datetime.now().isoformat()},
                timestamp=datetime.now()
            )
            
            self.validation_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating graph structure: {str(e)}")
            return ValidationResult(
                success=False,
                level=ValidationLevel.ERROR,
                error_code=ErrorCode.GRAPH_STRUCTURE,
                message=f"Structure validation error: {str(e)}",
                details={'error': traceback.format_exc()},
                timestamp=datetime.now()
            )

    def add_validation_rule(self, rule: ValidationRule):
        """Add a new validation rule."""
        if rule.name in self.rules:
            raise ValueError(f"Rule with name {rule.name} already exists")
            
        self.rules[rule.name] = rule
        self.logger.info(f"Added validation rule: {rule.name}")

    def _init_validation_rules(self):
        """Initialize default validation rules."""
        # Entity validation rules
        self.add_validation_rule(ValidationRule(
            name="required_entity_fields",
            description="Check required entity fields",
            level=ValidationLevel.ERROR,
            error_code=ErrorCode.MISSING_REQUIRED,
            validation_fn=self._validate_required_entity_fields,
            params={'required_fields': ['id', 'type']}
        ))
        
        self.add_validation_rule(ValidationRule(
            name="entity_id_format",
            description="Validate entity ID format",
            level=ValidationLevel.ERROR,
            error_code=ErrorCode.FORMAT_ERROR,
            validation_fn=self._validate_entity_id_format,
            params={'pattern': r'^[a-zA-Z0-9_-]+$'}
        ))
        
        # Relationship validation rules
        self.add_validation_rule(ValidationRule(
            name="required_relationship_fields",
            description="Check required relationship fields",
            level=ValidationLevel.ERROR,
            error_code=ErrorCode.MISSING_REQUIRED,
            validation_fn=self._validate_required_relationship_fields,
            params={'required_fields': ['source', 'target', 'type']}
        ))
        
        self.add_validation_rule(ValidationRule(
            name="relationship_nodes_exist",
            description="Validate relationship endpoints exist",
            level=ValidationLevel.ERROR,
            error_code=ErrorCode.RELATIONSHIP_ERROR,
            validation_fn=self._validate_relationship_nodes_exist,
            params={}
        ))
        
        # Structure validation rules
        self.add_validation_rule(ValidationRule(
            name="graph_connectivity",
            description="Check graph connectivity",
            level=ValidationLevel.WARNING,
            error_code=ErrorCode.GRAPH_STRUCTURE,
            validation_fn=self._validate_graph_connectivity,
            params={'min_connectivity': 0.8}
        ))

    def _apply_rule(self, rule: ValidationRule, data: Dict[str, Any]) -> ValidationResult:
        """Apply a validation rule to data."""
        try:
            is_valid = rule.validation_fn(data, rule.params)
            
            if is_valid:
                return ValidationResult(
                    success=True,
                    level=ValidationLevel.INFO,
                    error_code=None,
                    message=f"Validation rule '{rule.name}' passed",
                    details={},
                    timestamp=datetime.now()
                )
            else:
                return ValidationResult(
                    success=False,
                    level=rule.level,
                    error_code=rule.error_code,
                    message=f"Validation rule '{rule.name}' failed: {rule.description}",
                    details={'data': data, 'params': rule.params},
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            self.logger.error(f"Error applying validation rule {rule.name}: {str(e)}")
            return ValidationResult(
                success=False,
                level=ValidationLevel.ERROR,
                error_code=ErrorCode.INVALID_INPUT,
                message=f"Error applying validation rule: {str(e)}",
                details={'error': traceback.format_exc()},
                timestamp=datetime.now()
            )

    def _validate_required_entity_fields(self, entity: Dict[str, Any],
                                       params: Dict[str, Any]) -> bool:
        """Validate required entity fields are present."""
        required_fields = params['required_fields']
        return all(field in entity for field in required_fields)

    def _validate_entity_id_format(self, entity: Dict[str, Any],
                                 params: Dict[str, Any]) -> bool:
        """Validate entity ID format."""
        pattern = params['pattern']
        entity_id = entity.get('id', '')
        return bool(re.match(pattern, entity_id))

    def _validate_required_relationship_fields(self, relationship: Dict[str, Any],
                                            params: Dict[str, Any]) -> bool:
        """Validate required relationship fields are present."""
        required_fields = params['required_fields']
        return all(field in relationship for field in required_fields)

    def _validate_relationship_nodes_exist(self, relationship: Dict[str, Any],
                                         params: Dict[str, Any]) -> bool:
        """Validate both ends of relationship exist in graph."""
        query = """
        MATCH (s:Entity {id: $source})
        MATCH (t:Entity {id: $target})
        RETURN COUNT(*) as count
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(
                query,
                source=relationship['source'],
                target=relationship['target']
            )
            count = result.single()['count']
            return count == 2

    def _validate_graph_connectivity(self, data: Dict[str, Any],
                                   params: Dict[str, Any]) -> bool:
        """Validate graph connectivity meets minimum threshold."""
        min_connectivity = params['min_connectivity']
        
        query = """
        MATCH (n:Entity)
        WITH count(n) as total_nodes
        MATCH (n:Entity)-[r]-(m:Entity)
        WITH total_nodes, count(DISTINCT n) + count(DISTINCT m) as connected_nodes
        RETURN toFloat(connected_nodes) / total_nodes as connectivity
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            connectivity = result.single()['connectivity']
            return connectivity >= min_connectivity

    def _get_rules_by_type(self, rule_type: str) -> List[ValidationRule]:
        """Get validation rules for a specific type."""
        return [
            rule for rule in self.rules.values()
            if rule.name.startswith(f"{rule_type}_")
        ]

    def _get_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate cache key for validation results."""
        return f"{prefix}_{hash(json.dumps(data, sort_keys=True))}"