# database/pathfinding.py
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import heapq
import networkx as nx
from collections import defaultdict
import torch
import numpy as np

#from ..config import Config
from config import Config
from .graph import Neo4jConnection
from .analytics import GraphAnalyzer

@dataclass
class PathCost:
    """Container for path cost components."""
    distance: float  # Topological distance
    semantic_cost: float  # Semantic relevance cost
    confidence: float  # Path confidence
    total_cost: float  # Combined cost

@dataclass
class Path:
    """Container for a path in the graph."""
    nodes: List[str]
    relationships: List[str]
    cost: PathCost
    metadata: Dict[str, Any]

class PathFinder:
    """Handles intelligent path finding and query routing."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        self.graph_analyzer = GraphAnalyzer(config, neo4j_connection)
        
        # Path finding configurations
        self.path_config = {
            'max_paths': config.model_config.get('max_paths', 5),
            'max_path_length': config.model_config.get('max_path_length', 6),
            'min_confidence': config.model_config.get('min_confidence', 0.3),
            'semantic_weight': config.model_config.get('semantic_weight', 0.4),
            'diversity_weight': config.model_config.get('diversity_weight', 0.3)
        }

    def find_paths(self, start_node: str, end_node: str, 
                  query_embedding: Optional[torch.Tensor] = None) -> List[Path]:
        """
        Find multiple diverse paths between nodes.
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            query_embedding: Optional query embedding for semantic relevance
        """
        try:
            paths = []
            visited_paths = set()
            
            # Get initial path using A*
            initial_path = self._find_shortest_path(
                start_node,
                end_node,
                query_embedding
            )
            
            if initial_path:
                paths.append(initial_path)
                visited_paths.add(tuple(initial_path.nodes))
            
            # Find additional diverse paths
            while len(paths) < self.path_config['max_paths']:
                # Find next most diverse path
                next_path = self._find_diverse_path(
                    start_node,
                    end_node,
                    paths,
                    visited_paths,
                    query_embedding
                )
                
                if not next_path:
                    break
                    
                paths.append(next_path)
                visited_paths.add(tuple(next_path.nodes))
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error finding paths: {str(e)}")
            raise

    def find_multi_target_paths(self, start_node: str, target_nodes: List[str],
                              query_embedding: Optional[torch.Tensor] = None) -> List[Path]:
        """Find paths from start node to multiple targets."""
        try:
            paths = []
            visited_nodes = {start_node}
            remaining_targets = set(target_nodes)
            
            while remaining_targets and len(paths) < self.path_config['max_paths']:
                # Find closest unvisited target
                closest_target = None
                best_path = None
                best_cost = float('inf')
                
                for target in remaining_targets:
                    path = self._find_shortest_path(
                        start_node,
                        target,
                        query_embedding,
                        visited_nodes
                    )
                    
                    if path and path.cost.total_cost < best_cost:
                        closest_target = target
                        best_path = path
                        best_cost = path.cost.total_cost
                
                if not best_path:
                    break
                    
                paths.append(best_path)
                visited_nodes.update(best_path.nodes)
                remaining_targets.remove(closest_target)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error finding multi-target paths: {str(e)}")
            raise

    def find_subgraph_paths(self, nodes: List[str],
                           query_embedding: Optional[torch.Tensor] = None) -> List[Path]:
        """Find paths connecting a set of nodes into a minimal subgraph."""
        try:
            paths = []
            connected_nodes = set()
            
            # Sort nodes by importance
            node_importance = self.graph_analyzer.get_node_importance_ranking()
            node_scores = {n: s for n, s in node_importance}
            sorted_nodes = sorted(nodes, key=lambda n: node_scores.get(n, 0), reverse=True)
            
            # Start with most important node
            current_node = sorted_nodes[0]
            connected_nodes.add(current_node)
            
            # Iteratively connect remaining nodes
            while len(connected_nodes) < len(nodes):
                best_path = None
                best_cost = float('inf')
                best_target = None
                
                # Find best path from connected component to unconnected node
                for node in sorted_nodes:
                    if node in connected_nodes:
                        continue
                        
                    # Try to find path from any connected node
                    for start in connected_nodes:
                        path = self._find_shortest_path(
                            start,
                            node,
                            query_embedding
                        )
                        
                        if path and path.cost.total_cost < best_cost:
                            best_path = path
                            best_cost = path.cost.total_cost
                            best_target = node
                
                if not best_path:
                    break
                    
                paths.append(best_path)
                connected_nodes.add(best_target)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error finding subgraph paths: {str(e)}")
            raise

    def _find_shortest_path(self, start_node: str, end_node: str,
                          query_embedding: Optional[torch.Tensor] = None,
                          avoid_nodes: Optional[Set[str]] = None) -> Optional[Path]:
        """Find shortest path using A* search with custom cost function."""
        try:
            # Initialize data structures
            visited = set()
            if avoid_nodes:
                visited.update(avoid_nodes)
                
            pq = [(0, start_node, [], [])]  # (cost, node, path_nodes, path_rels)
            costs = {start_node: 0}  # g-scores in A*
            
            while pq:
                current_cost, current, path_nodes, path_rels = heapq.heappop(pq)
                
                if current == end_node:
                    # Path found
                    path_nodes.append(current)
                    
                    return Path(
                        nodes=path_nodes,
                        relationships=path_rels,
                        cost=self._calculate_path_cost(
                            path_nodes,
                            path_rels,
                            query_embedding
                        ),
                        metadata={
                            'length': len(path_nodes) - 1,
                            'avg_confidence': sum(self._get_edge_confidence(
                                path_nodes[i],
                                path_nodes[i+1]
                            ) for i in range(len(path_nodes)-1)) / (len(path_nodes)-1)
                        }
                    )
                
                if current in visited:
                    continue
                    
                visited.add(current)
                path_nodes.append(current)
                
                # Get neighbors
                neighbors = self._get_neighbors(current)
                
                for neighbor, rel_type in neighbors:
                    if neighbor in visited:
                        continue
                        
                    if len(path_nodes) >= self.path_config['max_path_length']:
                        continue
                    
                    # Calculate costs
                    g_cost = current_cost + self._calculate_edge_cost(
                        current,
                        neighbor,
                        rel_type,
                        query_embedding
                    )
                    
                    h_cost = self._estimate_remaining_cost(
                        neighbor,
                        end_node
                    )
                    
                    total_cost = g_cost + h_cost
                    
                    # Update if better path found
                    if neighbor not in costs or g_cost < costs[neighbor]:
                        costs[neighbor] = g_cost
                        new_path_nodes = path_nodes.copy()
                        new_path_rels = path_rels.copy()
                        new_path_rels.append(rel_type)
                        heapq.heappush(
                            pq,
                            (total_cost, neighbor, new_path_nodes, new_path_rels)
                        )
            
            return None  # No path found
            
        except Exception as e:
            self.logger.error(f"Error finding shortest path: {str(e)}")
            raise

    def _find_diverse_path(self, start_node: str, end_node: str,
                         existing_paths: List[Path], visited_paths: Set[Tuple[str]],
                         query_embedding: Optional[torch.Tensor] = None) -> Optional[Path]:
        """Find a diverse path different from existing ones."""
        try:
            # Initialize data structures
            pq = [(0, start_node, [], [])]  # (cost, node, path_nodes, path_rels)
            costs = {start_node: 0}
            
            while pq:
                current_cost, current, path_nodes, path_rels = heapq.heappop(pq)
                current_path = path_nodes + [current]
                
                if current == end_node:
                    # Check if path is sufficiently diverse
                    if self._is_path_diverse(
                        current_path,
                        existing_paths,
                        self.path_config['diversity_weight']
                    ):
                        path_cost = self._calculate_path_cost(
                            current_path,
                            path_rels,
                            query_embedding
                        )
                        
                        return Path(
                            nodes=current_path,
                            relationships=path_rels,
                            cost=path_cost,
                            metadata={
                                'length': len(current_path) - 1,
                                'diversity_score': self._calculate_path_diversity(
                                    current_path,
                                    existing_paths
                                )
                            }
                        )
                
                # Get neighbors
                neighbors = self._get_neighbors(current)
                
                for neighbor, rel_type in neighbors:
                    if tuple(current_path + [neighbor]) in visited_paths:
                        continue
                        
                    if len(current_path) >= self.path_config['max_path_length']:
                        continue
                    
                    # Calculate costs with diversity penalty
                    g_cost = current_cost + self._calculate_edge_cost(
                        current,
                        neighbor,
                        rel_type,
                        query_embedding
                    )
                    
                    # Add diversity penalty
                    diversity_penalty = self._calculate_diversity_penalty(
                        current_path + [neighbor],
                        existing_paths
                    )
                    
                    g_cost += diversity_penalty
                    
                    h_cost = self._estimate_remaining_cost(neighbor, end_node)
                    total_cost = g_cost + h_cost
                    
                    # Update if better path found
                    if neighbor not in costs or g_cost < costs[neighbor]:
                        costs[neighbor] = g_cost
                        new_path_nodes = current_path.copy()
                        new_path_rels = path_rels.copy()
                        new_path_rels.append(rel_type)
                        heapq.heappush(
                            pq,
                            (total_cost, neighbor, new_path_nodes, new_path_rels)
                        )
            
            return None  # No diverse path found
            
        except Exception as e:
            self.logger.error(f"Error finding diverse path: {str(e)}")
            raise

    def _get_neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """Get neighboring nodes and relationship types."""
        query = """
        MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)
        RETURN m.id as neighbor_id, type(r) as rel_type
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_id=node_id)
            return [(r['neighbor_id'], r['rel_type']) for r in result]

    def _calculate_edge_cost(self, source: str, target: str,
                           rel_type: str, 
                           query_embedding: Optional[torch.Tensor]) -> float:
        """Calculate cost of traversing an edge."""
        # Base cost
        cost = 1.0
        
        # Add semantic cost if query embedding available
        if query_embedding is not None:
            semantic_cost = self._calculate_semantic_cost(
                source,
                target,
                rel_type,
                query_embedding
            )
            cost += self.path_config['semantic_weight'] * semantic_cost
        
        # Add confidence-based cost
        confidence = self._get_edge_confidence(source, target)
        confidence_cost = 1.0 - confidence
        cost += (1.0 - self.path_config['semantic_weight']) * confidence_cost
        
        return cost

    def _calculate_semantic_cost(self, source: str, target: str,
                               rel_type: str,
                               query_embedding: torch.Tensor) -> float:
        """Calculate semantic relevance cost."""
        # Get embeddings
        source_embedding = self._get_node_embedding(source)
        target_embedding = self._get_node_embedding(target)
        rel_embedding = self._get_relationship_embedding(rel_type)
        
        # Calculate similarity with query
        source_sim = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            source_embedding.unsqueeze(0)
        ).item()
        
        target_sim = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            target_embedding.unsqueeze(0)
        ).item()
        
        rel_sim = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            rel_embedding.unsqueeze(0)
        ).item()
        
        # Combine similarities
        avg_sim = (source_sim + target_sim + rel_sim) / 3
        return 1.0 - avg_sim

    def _get_edge_confidence(self, source: str, target: str) -> float:
        """Get confidence score for edge."""
        query = """
        MATCH (n:Entity {id: $source})-[r]-(m:Entity {id: $target})
        RETURN r.confidence as confidence
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, source=source, target=target)
            record = result.single()
            return record['confidence'] if record else 0.0

    def _estimate_remaining_cost(self, current: str, target: str) -> float:
        """Estimate remaining cost (heuristic for A*)."""
        # Use pre-computed shortest path distances if available
        query = """
        MATCH (n:Entity {id: $current}), (m:Entity {id: $target})
        RETURN n.x as n_x, n.y as n_y, m.x as m_x, m.y as m_y
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, current=current, target=target)
            record = result.single()
            
            if record and all(v is not None for v in record.values()):
                # Use euclidean distance in embedded space
                return np.sqrt(
                    (record['n_x'] - record['m_x'])**2 +
                    (record['n_y'] - record['m_y'])**2
                )
            
            # Fallback to topology-based estimate
            return self._estimate_topological_distance(current, target)

    def _estimate_topological_distance(self, source: str, target: str) -> float:
        """Estimate topological distance between nodes."""
        query = """
        MATCH (n:Entity {id: $source})-[*..2]-(m:Entity)
        WITH collect(m.id) as neighbors
        RETURN $target IN neighbors as is_close
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, source=source, target=target)
            record = result.single()
            
            # If target is within 2 hops, use actual distance
            if record and record['is_close']:
                return 2.0
            
            # Otherwise estimate based on graph metrics
            return self.path_config['max_path_length'] / 2

    def _calculate_path_cost(self, nodes: List[str], relationships: List[str],
                           query_embedding: Optional[torch.Tensor]) -> PathCost:
        """Calculate complete cost metrics for a path."""
        # Calculate distance cost
        distance = len(nodes) - 1
        
        # Calculate semantic cost if query provided
        if query_embedding is not None:
            semantic_costs = []
            for i in range(len(nodes) - 1):
                semantic_cost = self._calculate_semantic_cost(
                    nodes[i],
                    nodes[i+1],
                    relationships[i],
                    query_embedding
                )
                semantic_costs.append(semantic_cost)
            avg_semantic_cost = sum(semantic_costs) / len(semantic_costs)
        else:
            avg_semantic_cost = 0.0
        
        # Calculate confidence
        confidences = []
        for i in range(len(nodes) - 1):
            confidence = self._get_edge_confidence(nodes[i], nodes[i+1])
            confidences.append(confidence)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Calculate total cost
        total_cost = (
            distance +
            self.path_config['semantic_weight'] * avg_semantic_cost +
            (1.0 - self.path_config['semantic_weight']) * (1.0 - avg_confidence)
        )
        
        return PathCost(
            distance=float(distance),
            semantic_cost=float(avg_semantic_cost),
            confidence=float(avg_confidence),
            total_cost=float(total_cost)
        )

    def _is_path_diverse(self, path: List[str], existing_paths: List[Path],
                        threshold: float) -> bool:
        """Check if path is sufficiently diverse from existing paths."""
        if not existing_paths:
            return True
            
        diversity_score = self._calculate_path_diversity(path, existing_paths)
        return diversity_score >= threshold

    def _calculate_path_diversity(self, path: List[str],
                                existing_paths: List[Path]) -> float:
        """Calculate diversity score of a path compared to existing paths."""
        if not existing_paths:
            return 1.0
            
        path_set = set(path)
        
        # Calculate Jaccard distances to existing paths
        distances = []
        for existing in existing_paths:
            existing_set = set(existing.nodes)
            intersection = len(path_set.intersection(existing_set))
            union = len(path_set.union(existing_set))
            distance = 1.0 - (intersection / union if union > 0 else 0.0)
            distances.append(distance)
        
        # Return minimum distance (maximum similarity)
        return min(distances)

    def _calculate_diversity_penalty(self, path: List[str],
                                  existing_paths: List[Path]) -> float:
        """Calculate penalty for path similarity to existing paths."""
        diversity_score = self._calculate_path_diversity(path, existing_paths)
        return (1.0 - diversity_score) * self.path_config['diversity_weight']

    def _get_node_embedding(self, node_id: str) -> torch.Tensor:
        """Get pre-computed node embedding."""
        query = """
        MATCH (n:Entity {id: $node_id})
        RETURN n.embedding as embedding
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if record and record['embedding']:
                return torch.tensor(record['embedding'])
            
            # Return zero embedding if not found
            return torch.zeros(self.config.model_config.get('embedding_dim', 768))

    def _get_relationship_embedding(self, rel_type: str) -> torch.Tensor:
        """Get pre-computed relationship type embedding."""
        query = """
        MATCH ()-[r:$rel_type]->()
        RETURN r.embedding as embedding LIMIT 1
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, rel_type=rel_type)
            record = result.single()
            
            if record and record['embedding']:
                return torch.tensor(record['embedding'])
            
            # Return zero embedding if not found
            return torch.zeros(self.config.model_config.get('embedding_dim', 768))