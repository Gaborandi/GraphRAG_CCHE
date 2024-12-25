# config.py
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import yaml

class Config:
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config: Dict = self._load_config()
        
        # Model configurations
        self.model_name = self.config.get('model', {}).get('name', 'meta-llama/Llama-3.2-1B')
        self.model_config = self.config.get('model', {})
        
        # Document processing configurations
        self.chunk_size = self.config.get('processing', {}).get('chunk_size', 1000)
        self.overlap = self.config.get('processing', {}).get('overlap', 200)
        
        # Neo4j configurations
        self.neo4j_config = self.config.get('neo4j', {})
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            filename=log_config.get('file', 'knowledge_graph.log')
        )

# Example config.yaml structure:
"""
model:
  name: meta-llama/Llama-3.2-1B
  batch_size: 8
  max_length: 512

processing:
  chunk_size: 1000
  overlap: 200

neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: password

logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  file: knowledge_graph.log
"""