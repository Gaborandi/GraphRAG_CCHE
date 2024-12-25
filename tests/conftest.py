# tests/conftest.py
import pytest
from pathlib import Path
import tempfile
import yaml
from typing import Dict, Any

from ..config import Config
from ..database.graph import Neo4jConnection
from ..llm.model import LlamaProcessor
from ..orchestration.pipeline import KnowledgeGraphPipeline

@pytest.fixture
def test_config() -> Config:
    """Create test configuration."""
    config_data = {
        'model': {
            'name': 'meta-llama/Llama-3.2-1B',
            'batch_size': 4,
            'max_length': 512
        },
        'processing': {
            'chunk_size': 1000,
            'overlap': 200
        },
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'test_password'
        },
        'logging': {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        return Config(f.name)

@pytest.fixture
def test_documents(tmp_path: Path) -> Dict[str, Path]:
    """Create test documents."""
    documents = {}
    
    # Create test PDF
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\ntest content")
    documents['pdf'] = pdf_path
    
    # Create test DOCX
    docx_path = tmp_path / "test.docx"
    docx_path.write_text("Test document content")
    documents['docx'] = docx_path
    
    # Create test CSV
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("id,name,value\n1,test,100")
    documents['csv'] = csv_path
    
    return documents

