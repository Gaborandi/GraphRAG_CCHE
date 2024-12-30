# database/retrieval.py
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
from queue import PriorityQueue
import numpy as np
from datetime import datetime

#from ..config import Config
from config import Config
from .community import Community, CommunityDetector
from .graph import Neo4jConnection
from .q_learning import QValueTrainer

@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    subgraphs: List[Dict[str, Any]]
    communities: List[Community]
    search_path: List[str]
    confidence: float
    retrieval_time: float

class QValueEstimator(nn.Module):
    """Neural network for Q-value estimation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class GraphRetriever:
    """Handles graph-guided retrieval with Q-value optimization."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        self.community_detector = CommunityDetector(config, neo4j_connection)
        
        # Initialize Q-value estimator
        self.q_estimator = QValueEstimator(input_dim=config.model_config.get('embedding_dim', 768))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_estimator.to(self.device)
        
        self.vector_index = FaissIndex(config.vector_dim)
        self.query_optimizer = QueryOptimizer()
        self.result_merger = ResultMerger()

    async def hybrid_search(self, query, embedding):
        # Run searches in parallel
        [graph_results, vector_results] = await asyncio.gather(
            self.graph_search(query),
            self.vector_search(embedding)
        )
        
        # Merge and rank results
        merged_results = self.result_merger.merge(
            graph_results,
            vector_results,
            strategy='reciprocal_rank_fusion'
        )
        return merged_results

    def retrieve(self, query: str, embedding_model, 
                max_subgraphs: int = 5, max_depth: int = 3) -> RetrievalResult:
        """
        Retrieve relevant subgraphs using graph-guided search.
        Args:
            query: Search query
            embedding_model: Model for text embeddings
            max_subgraphs: Maximum number of subgraphs to return
            max_depth: Maximum search depth
        """
        start_time = datetime.now()
        
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query, embedding_model)
            
            # Get initial communities based on query similarity
            relevant_communities = self._get_relevant_communities(
                query_embedding, 
                max_subgraphs
            )
            
            # Initialize search structures
            search_queue = PriorityQueue()
            visited_nodes = set()
            search_path = []
            
            # Initialize with community centers
            for community in relevant_communities:
                center_node = self._get_community_center(community)
                if center_node:
                    priority = self._estimate_q_value(
                        query_embedding,
                        self._get_node_embedding(center_node, embedding_model)
                    )
                    search_queue.put((-priority, center_node))
            
            # Perform graph-guided search
            subgraphs = []
            while not search_queue.empty() and len(subgraphs) < max_subgraphs:
                # Get highest priority node
                priority, current_node = search_queue.get()
                
                if current_node in visited_nodes:
                    continue
                    
                visited_nodes.add(current_node)
                search_path.append(current_node)
                
                # Extract subgraph around current node
                subgraph = self._extract_subgraph(current_node, max_depth)
                if subgraph:
                    subgraphs.append(subgraph)
                
                # Add neighbors to queue
                neighbors = self._get_node_neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        neighbor_embedding = self._get_node_embedding(
                            neighbor, 
                            embedding_model
                        )
                        priority = self._estimate_q_value(
                            query_embedding,
                            neighbor_embedding
                        )
                        search_queue.put((-priority, neighbor))
            
            # Calculate overall confidence
            confidence = self._calculate_retrieval_confidence(subgraphs)
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            return RetrievalResult(
                subgraphs=subgraphs,
                communities=relevant_communities,
                search_path=search_path,
                confidence=confidence,
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in graph retrieval: {str(e)}")
            raise

    def _get_query_embedding(self, query: str, 
                           embedding_model) -> torch.Tensor:
        """Get embedding for query text."""
        with torch.no_grad():
            embedding = embedding_model.encode(query)
            return torch.tensor(embedding).to(self.device)

    def _get_relevant_communities(self, query_embedding: torch.Tensor, 
                                max_communities: int) -> List[Community]:
        """Get most relevant communities based on query."""
        communities = self.community_detector.detect_communities()
        
        # Score communities based on summary relevance
        community_scores = []
        for community in communities:
            if community.summary:
                summary_embedding = self._get_text_embedding(
                    community.summary,
                    embedding_model
                )
                score = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    summary_embedding.unsqueeze(0)
                ).item()
                community_scores.append((score, community))
        
        # Sort and return top communities
        community_scores.sort(reverse=True)
        return [c for _, c in community_scores[:max_communities]]

    def _get_community_center(self, community: Community) -> Optional[str]:
        """Get central node of a community based on eigenvector centrality."""
        query = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.id IN $node_ids AND m.id IN $node_ids
        WITH n, count(r) as degree
        RETURN n.id
        ORDER BY degree DESC
        LIMIT 1
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_ids=list(community.nodes))
            record = result.single()
            return record[0] if record else None

    def _estimate_q_value(self, query_embedding: torch.Tensor, 
                         node_embedding: torch.Tensor) -> float:
        """Estimate Q-value for a node given the query."""
        with torch.no_grad():
            # Combine query and node embeddings
            combined = torch.cat([query_embedding, node_embedding])
            q_value = self.q_estimator(combined)
            return q_value.item()

    def _extract_subgraph(self, center_node: str, max_depth: int) -> Dict[str, Any]:
        """Extract subgraph around a center node."""
        query = """
        MATCH path = (center:Entity {id: $center_id})-[*1..$max_depth]-(n:Entity)
        RETURN path
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, center_id=center_node, max_depth=max_depth)
            
            # Process paths into subgraph
            subgraph = {
                'nodes': set(),
                'edges': set(),
                'center': center_node
            }
            
            for record in result:
                path = record['path']
                # Add nodes and edges from path
                for node in path.nodes:
                    subgraph['nodes'].add(node['id'])
                for rel in path.relationships:
                    subgraph['edges'].add((
                        rel.start_node['id'],
                        rel.type,
                        rel.end_node['id']
                    ))
            
            return subgraph if subgraph['nodes'] else None

    def _get_node_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes."""
        query = """
        MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)
        RETURN m.id
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_id=node_id)
            return [record[0] for record in result]

    def _calculate_retrieval_confidence(self, 
                                     subgraphs: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in retrieval results."""
        if not subgraphs:
            return 0.0
            
        # Use subgraph size and connectivity as confidence indicators
        scores = []
        for subgraph in subgraphs:
            size_score = min(1.0, len(subgraph['nodes']) / 100)
            density_score = len(subgraph['edges']) / (len(subgraph['nodes']) * (len(subgraph['nodes']) - 1))
            scores.append((size_score + density_score) / 2)
            
        return sum(scores) / len(scores)

    def update_q_estimator(self, training_data: List[Tuple[str, str, float]]) -> None:
        """Update Q-value estimator with new training data."""
        self.q_estimator.train()
        
        optimizer = torch.optim.Adam(self.q_estimator.parameters())
        criterion = nn.MSELoss()
        
        for query, node_id, target_q in training_data:
            try:
                # Get embeddings
                query_embedding = self._get_query_embedding(query, embedding_model)
                node_embedding = self._get_node_embedding(node_id, embedding_model)
                
                # Combine embeddings
                combined = torch.cat([query_embedding, node_embedding])
                
                # Forward pass
                predicted_q = self.q_estimator(combined)
                target_q = torch.tensor([target_q]).to(self.device)
                
                # Compute loss and update
                loss = criterion(predicted_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                self.logger.error(f"Error updating Q-estimator: {str(e)}")
                continue
        
        self.q_estimator.eval()

    def save_q_estimator(self, path: str) -> None:
        """Save Q-value estimator model."""
        torch.save(self.q_estimator.state_dict(), path)

    def load_q_estimator(self, path: str) -> None:
        """Load Q-value estimator model."""
        self.q_estimator.load_state_dict(torch.load(path))
        self.q_estimator.eval()

    def train_q_network(self, embedding_model, num_episodes: int = 1000) -> Dict[str, List[float]]:
        """Train Q-network for better path selection."""
        trainer = QValueTrainer(self.config, self.neo4j)
        metrics = trainer.train(embedding_model, num_episodes)
        self.q_trainer = trainer
        return metrics
    
    def _estimate_q_value(self, query_embedding: torch.Tensor, 
                         node_embedding: torch.Tensor) -> float:
        """Estimate Q-value using trained network."""
        if hasattr(self, 'q_trainer'):
            return self.q_trainer.predict_q_value(query_embedding, node_embedding)
        return super()._estimate_q_value(query_embedding, node_embedding)