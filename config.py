# config.py
from pathlib import Path
import logging
from typing import Dict, Any, Union
import yaml

class Config:
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load_config()
        
        # Model configurations
        self.model_name = self.config.get('model', {}).get('name', 'meta-llama/Llama-3.2-1B')
        self.model_config = self.config.get('model', {})
        
        # Document processing configurations
        self.chunk_size = self.config.get('processing', {}).get('chunk_size', 1000)
        self.overlap = self.config.get('processing', {}).get('overlap', 200)
        
        # Neo4j configurations (if needed)
        self.neo4j_config = self.config.get('neo4j', {})
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration based on YAML content."""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            filename=log_config.get('file', 'knowledge_graph.log')
        )
