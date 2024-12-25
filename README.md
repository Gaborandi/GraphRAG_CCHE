# docs/README.md
# Knowledge Graph Creation System

A robust system for creating knowledge graphs from documents using Llama and Neo4j.

## Features

- Multi-format document processing (PDF, DOCX, CSV)
- Entity and relationship extraction using Llama
- Knowledge graph storage in Neo4j
- Parallel processing capabilities
- Comprehensive error handling
- CLI interface

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/knowledge-graph-system.git

# Install dependencies
pip install -r requirements.txt

# Setup Neo4j
docker-compose -f deployment/docker/docker-compose.yml up -d
```

## Usage

```bash
# Process documents
python -m orchestration.cli process config.yaml /path/to/documents

# Query graph
python -m orchestration.cli query config.yaml --entity "John Smith"
```

## Configuration

Create a `config.yaml` file:

```yaml
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
  password: your_password
```

# docs/API.md
# API Documentation

## Document Processing

### DocumentProcessor

Base class for document processors.

```python
from document_processor.base import DocumentProcessor

class CustomProcessor(DocumentProcessor):
    def process(self, file_path: Path) -> Document:
        # Implementation
        pass
```

### LlamaProcessor

Handles text processing and information extraction.

```python
from llm.model import LlamaProcessor

processor = LlamaProcessor(config)
results = processor.process_document(document)
```

### KnowledgeGraph

Manages Neo4j graph operations.

```python
from database.graph import KnowledgeGraph

graph = KnowledgeGraph(config)
graph.process_extraction_results(doc_id, results)
```

## Testing

Run tests:

```bash
pytest tests/
```

## Deployment

Deploy with Docker:

```bash
cd deployment/docker
docker-compose up -d
```