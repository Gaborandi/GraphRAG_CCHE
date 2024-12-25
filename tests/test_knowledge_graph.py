# tests/test_knowledge_graph.py
import pytest
from ..database.graph import KnowledgeGraph

def test_knowledge_graph(test_config):
    """Test knowledge graph operations."""
    graph = KnowledgeGraph(test_config)
    
    # Test entity creation
    entities = [
        {'text': 'John Smith', 'type': 'PERSON'},
        {'text': 'Apple Inc.', 'type': 'ORGANIZATION'}
    ]
    
    result = ExtractionResult(
        entities=entities,
        relationships=[{
            'source': 'John Smith',
            'relationship': 'works_for',
            'target': 'Apple Inc.'
        }],
        confidence=0.9,
        chunk_id='test_chunk'
    )
    
    graph.process_extraction_results('test_doc', [result])
    
    # Query and verify
    stored_entities = graph.query_entities()
    assert len(stored_entities) >= 2

# deployment/docker/Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create volume for configurations
VOLUME /app/config

# Default command
CMD ["python", "-m", "orchestration.cli"]

# deployment/docker/docker-compose.yml
version: '3.8'

services:
  knowledge_graph:
    build: .
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:4.4
    environment:
      - NEO4J_AUTH=neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  neo4j_data:
  neo4j_logs:
