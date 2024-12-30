# database/analytics.py
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import numpy as np
from collections import defaultdict
import community
import json

#from ..config import Config
from config import Config
from .graph import Neo4jConnection

@dataclass
class NodeMetrics:
    """Container for node-level metrics."""
    node_id: str
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    clustering_coefficient: float
    core_number: int
    community_id: Optional[int] = None

@dataclass
class GraphMetrics:
    """Container for graph-level metrics."""
    total_nodes: int
    total_edges: int
    density: float
    average_clustering: float
    average_path_length: float
    diameter: float
    num_connected_components: int
    modularity: float
    assortativity: float
    node_type_distribution: Dict[str, int]
    relationship_type_distribution: Dict[str, int]

class GraphAnalyzer:
    """Handles advanced graph analytics and metrics calculation."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for computed metrics
        self.node_metrics_cache: Dict[str, NodeMetrics] = {}
        self.graph_metrics_cache: Optional[GraphMetrics] = None
        self.last_update: Optional[datetime] = None
        
        # Cache validity period (in seconds)
        self.cache_validity = config.model_config.get('cache_validity', 3600)

    def compute_node_metrics(self, node_ids: Optional[List[str]] = None) -> Dict[str, NodeMetrics]:
        """
        Compute centrality and importance metrics for nodes.
        Args:
            node_ids: Optional list of specific nodes to analyze. If None, analyze all nodes.
        """
        try:
            # Check cache validity
            if self._is_cache_valid() and not node_ids:
                return self.node_metrics_cache
            
            # Get graph data
            G = self._get_networkx_graph(node_ids)
            
            # Compute basic centrality metrics
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            eigenvector_cent = nx.eigenvector_centrality_numpy(G)
            pagerank = nx.pagerank(G)
            
            # Compute clustering coefficients
            clustering = nx.clustering(G)
            
            # Compute k-core decomposition
            core_numbers = nx.core_number(G)
            
            # Detect communities
            communities = community.best_partition(G)
            
            # Create metrics for each node
            node_metrics = {}
            for node in G.nodes():
                metrics = NodeMetrics(
                    node_id=node,
                    degree_centrality=degree_cent[node],
                    betweenness_centrality=betweenness_cent[node],
                    eigenvector_centrality=eigenvector_cent[node],
                    pagerank=pagerank[node],
                    clustering_coefficient=clustering[node],
                    core_number=core_numbers[node],
                    community_id=communities[node]
                )
                node_metrics[node] = metrics
            
            # Update cache if computing for all nodes
            if not node_ids:
                self.node_metrics_cache = node_metrics
                self.last_update = datetime.now()
            
            return node_metrics
            
        except Exception as e:
            self.logger.error(f"Error computing node metrics: {str(e)}")
            raise

    def compute_graph_metrics(self) -> GraphMetrics:
        """Compute graph-level metrics."""
        try:
            # Check cache validity
            if self._is_cache_valid() and self.graph_metrics_cache:
                return self.graph_metrics_cache
            
            # Get graph data
            G = self._get_networkx_graph()
            
            # Basic metrics
            total_nodes = G.number_of_nodes()
            total_edges = G.number_of_edges()
            density = nx.density(G)
            
            # Clustering
            avg_clustering = nx.average_clustering(G)
            
            # Path metrics (compute on largest component if disconnected)
            largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(largest_subgraph)
            diameter = nx.diameter(largest_subgraph)
            
            # Component analysis
            num_components = nx.number_connected_components(G)
            
            # Community detection and modularity
            communities = community.best_partition(G)
            modularity = community.modularity(communities, G)
            
            # Assortativity
            assortativity = nx.degree_assortativity_coefficient(G)
            
            # Node and relationship type distributions
            node_types = self._get_node_type_distribution()
            relationship_types = self._get_relationship_type_distribution()
            
            # Create metrics object
            metrics = GraphMetrics(
                total_nodes=total_nodes,
                total_edges=total_edges,
                density=density,
                average_clustering=avg_clustering,
                average_path_length=avg_path_length,
                diameter=diameter,
                num_connected_components=num_components,
                modularity=modularity,
                assortativity=assortativity,
                node_type_distribution=node_types,
                relationship_type_distribution=relationship_types
            )
            
            # Update cache
            self.graph_metrics_cache = metrics
            self.last_update = datetime.now()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing graph metrics: {str(e)}")
            raise

    def find_important_patterns(self) -> Dict[str, Any]:
        """Identify important patterns in the graph."""
        try:
            G = self._get_networkx_graph()
            patterns = {}
            
            # Find hubs (high-degree nodes)
            patterns['hubs'] = self._find_hubs(G)
            
            # Find structural holes (nodes bridging communities)
            patterns['bridges'] = self._find_bridges(G)
            
            # Find dense subgraphs
            patterns['dense_subgraphs'] = self._find_dense_subgraphs(G)
            
            # Find frequent motifs
            patterns['motifs'] = self._find_motifs(G)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding patterns: {str(e)}")
            raise

    def get_node_importance_ranking(self, 
                                  importance_type: str = 'combined') -> List[Tuple[str, float]]:
        """
        Get ranking of nodes by importance.
        Args:
            importance_type: Type of importance metric ('degree', 'betweenness', 
                           'eigenvector', 'pagerank', or 'combined')
        """
        try:
            metrics = self.compute_node_metrics()
            
            if importance_type == 'degree':
                scores = [(n, m.degree_centrality) for n, m in metrics.items()]
            elif importance_type == 'betweenness':
                scores = [(n, m.betweenness_centrality) for n, m in metrics.items()]
            elif importance_type == 'eigenvector':
                scores = [(n, m.eigenvector_centrality) for n, m in metrics.items()]
            elif importance_type == 'pagerank':
                scores = [(n, m.pagerank) for n, m in metrics.items()]
            else:  # combined
                scores = [
                    (n, (m.degree_centrality + m.betweenness_centrality + 
                         m.eigenvector_centrality + m.pagerank) / 4)
                    for n, m in metrics.items()
                ]
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores
            
        except Exception as e:
            self.logger.error(f"Error ranking nodes: {str(e)}")
            raise

    def get_community_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get metrics for each community."""
        try:
            # Get node metrics which includes community assignments
            node_metrics = self.compute_node_metrics()
            
            # Group nodes by community
            communities = defaultdict(list)
            for node_id, metrics in node_metrics.items():
                if metrics.community_id is not None:
                    communities[metrics.community_id].append(node_id)
            
            # Compute metrics for each community
            community_metrics = {}
            for comm_id, nodes in communities.items():
                subgraph = self._get_networkx_graph(nodes)
                
                metrics = {
                    'size': len(nodes),
                    'density': nx.density(subgraph),
                    'avg_clustering': nx.average_clustering(subgraph),
                    'diameter': nx.diameter(subgraph) if nx.is_connected(subgraph) else float('inf'),
                    'hub_nodes': self._find_hubs(subgraph, top_k=3)
                }
                
                community_metrics[comm_id] = metrics
            
            return community_metrics
            
        except Exception as e:
            self.logger.error(f"Error computing community metrics: {str(e)}")
            raise

    def _get_networkx_graph(self, node_ids: Optional[List[str]] = None) -> nx.Graph:
        """Get NetworkX graph from Neo4j."""
        # Build query
        if node_ids:
            query = """
            MATCH (n:Entity)-[r]-(m:Entity)
            WHERE n.id IN $node_ids AND m.id IN $node_ids
            RETURN n.id as source, type(r) as rel_type, m.id as target
            """
            params = {'node_ids': node_ids}
        else:
            query = """
            MATCH (n:Entity)-[r]-(m:Entity)
            RETURN n.id as source, type(r) as rel_type, m.id as target
            """
            params = {}
        
        # Create graph
        G = nx.Graph()
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, **params)
            
            for record in result:
                G.add_edge(
                    record['source'],
                    record['target'],
                    type=record['rel_type']
                )
        
        return G

    def _get_node_type_distribution(self) -> Dict[str, int]:
        """Get distribution of node types."""
        query = """
        MATCH (n:Entity)
        RETURN n.type as type, count(*) as count
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            return {r['type']: r['count'] for r in result}

    def _get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            return {r['type']: r['count'] for r in result}

    def _find_hubs(self, G: nx.Graph, top_k: int = 10) -> List[str]:
        """Find hub nodes based on degree centrality."""
        degree_cent = nx.degree_centrality(G)
        sorted_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:top_k]]

    def _find_bridges(self, G: nx.Graph) -> List[str]:
        """Find bridge nodes (high betweenness centrality)."""
        betweenness_cent = nx.betweenness_centrality(G)
        # Consider nodes in top 10% as bridges
        threshold = np.percentile(list(betweenness_cent.values()), 90)
        return [node for node, cent in betweenness_cent.items() if cent >= threshold]

    def _find_dense_subgraphs(self, G: nx.Graph, min_size: int = 3) -> List[Set[str]]:
        """Find dense subgraphs using k-clique communities."""
        # Use k-clique-communities method
        k = min_size
        dense_subgraphs = list(nx.community.k_clique_communities(G, k))
        return [set(subgraph) for subgraph in dense_subgraphs]

    def _find_motifs(self, G: nx.Graph) -> Dict[str, int]:
        """Find frequent motifs (small subgraph patterns)."""
        motifs = defaultdict(int)
        
        # Look for common motifs of size 3
        for n1 in G.nodes():
            neighbors = set(G.neighbors(n1))
            for n2 in neighbors:
                for n3 in neighbors:
                    if n2 < n3:  # Avoid counting twice
                        # Check connection between n2 and n3
                        if G.has_edge(n2, n3):
                            motifs['triangle'] += 1
                        else:
                            motifs['v_shape'] += 1
        
        return dict(motifs)

    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid."""
        if not self.last_update:
            return False
        
        age = (datetime.now() - self.last_update).total_seconds()
        return age < self.cache_validity