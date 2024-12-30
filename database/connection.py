# database/connection.py
import logging
from neo4j import GraphDatabase
from config import Config

logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Manages Neo4j database connection and operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logger
        self._driver = GraphDatabase.driver(
            config.neo4j_config['uri'],
            auth=(config.neo4j_config['user'], config.neo4j_config['password'])
        )
        
        self._init_database()

    def _init_database(self):
        """Initialize database schema and constraints."""
        with self._driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.text)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.error(f"Error creating constraint: {str(e)}")

    def close(self):
        """Close the database connection."""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
