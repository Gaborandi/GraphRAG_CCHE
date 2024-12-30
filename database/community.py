# database/community.py
from typing import Dict, List, Set, Any, Optional
import networkx as nx
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np
from sklearn.cluster import SpectralClustering
#from ..config import Config
from config import Config
#from .graph import Neo4jConnection
from database.connection import Neo4jConnection

@dataclass
class Community:
    """Represents a detected community in the graph."""
    id: str
    nodes: Set[str]
    metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    summary: Optional[str] = None

class CommunityDetector:
    def __init__(self, config: Config, neo4j_connection: Optional[Neo4jConnection] = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection = neo4j_connection or Neo4jConnection(config)
        
    def detect_communities(self, min_community_size: int = 3) -> List[Community]:
        """Detect communities in the graph using spectral clustering."""
        try:
            # Get graph data from Neo4j
            graph_data = self._get_graph_data()
            
            # Convert to NetworkX graph
            G = self._create_networkx_graph(graph_data)
            
            # Perform spectral clustering
            communities = self._perform_spectral_clustering(G, min_community_size)
            
            # Create Community objects
            return self._create_community_objects(communities, G)
            
        except Exception as e:
            self.logger.error(f"Error in community detection: {str(e)}")
            raise

    def _get_graph_data(self) -> Dict[str, Any]:
        """Retrieve graph data from Neo4j."""
        query = """
        MATCH (n)-[r]->(m)
        RETURN 
            n.id as source_id,
            n.text as source_text,
            n.type as source_type,
            type(r) as relationship_type,
            r.confidence as confidence,
            m.id as target_id,
            m.text as target_text,
            m.type as target_type
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            return result.data()

    def _create_networkx_graph(self, graph_data: List[Dict[str, Any]]) -> nx.Graph:
        """Create NetworkX graph from Neo4j data."""
        G = nx.Graph()
        
        for record in graph_data:
            # Add nodes with attributes
            G.add_node(record['source_id'], 
                      text=record['source_text'],
                      type=record['source_type'])
            G.add_node(record['target_id'], 
                      text=record['target_text'],
                      type=record['target_type'])
            
            # Add edge with weight based on confidence
            G.add_edge(record['source_id'], 
                      record['target_id'],
                      weight=record['confidence'],
                      type=record['relationship_type'])
        
        return G

    def _perform_spectral_clustering(self, G: nx.Graph, min_size: int) -> List[Set[str]]:
        """Perform spectral clustering on the graph."""
        if len(G) < 2:
            return [set(G.nodes())]
        
        # Create adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        
        # Estimate number of communities based on eigenvalues
        n_clusters = self._estimate_n_clusters(adj_matrix)
        
        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        cluster_labels = clustering.fit_predict(adj_matrix)
        
        # Group nodes into communities
        communities = []
        for i in range(n_clusters):
            community = set()
            for node_idx, label in enumerate(cluster_labels):
                if label == i:
                    node_id = list(G.nodes())[node_idx]
                    community.add(node_id)
            
            if len(community) >= min_size:
                communities.append(community)
        
        return communities

    def _estimate_n_clusters(self, adj_matrix: np.ndarray) -> int:
        """Estimate optimal number of clusters using eigenvalue analysis."""
        eigenvalues = np.linalg.eigvals(adj_matrix)
        eigenvalues = sorted(abs(eigenvalues), reverse=True)
        
        # Find elbow point in eigenvalue curve
        diffs = np.diff(eigenvalues)
        elbow = np.argmax(diffs) + 1
        
        # Ensure reasonable number of clusters
        return max(2, min(elbow, int(np.sqrt(len(adj_matrix)))))

    def _create_community_objects(self, communities: List[Set[str]], 
                                G: nx.Graph) -> List[Community]:
        """Create Community objects from detected communities."""
        community_objects = []
        
        for idx, nodes in enumerate(communities):
            # Calculate community metadata
            metadata = self._calculate_community_metadata(nodes, G)
            
            community = Community(
                id=f"comm_{idx}",
                nodes=nodes,
                metadata=metadata,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            community_objects.append(community)
        
        return community_objects

    def _calculate_community_metadata(self, nodes: Set[str], 
                                   G: nx.Graph) -> Dict[str, Any]:
        """Calculate metadata for a community."""
        subgraph = G.subgraph(nodes)
        
        metadata = {
            'size': len(nodes),
            'density': nx.density(subgraph),
            'avg_degree': sum(dict(subgraph.degree()).values()) / len(nodes),
            'node_types': self._count_node_types(subgraph),
            'edge_types': self._count_edge_types(subgraph)
        }
        
        return metadata

    def _count_node_types(self, G: nx.Graph) -> Dict[str, int]:
        """Count frequency of each node type in the graph."""
        type_counts = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('type')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts

    def _count_edge_types(self, G: nx.Graph) -> Dict[str, int]:
        """Count frequency of each edge type in the graph."""
        type_counts = {}
        for _, _, data in G.edges(data=True):
            edge_type = data.get('type')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts

    def update_community_summaries(self, llm_processor) -> None:
        """Update summaries for all communities using LLM."""
        communities = self.detect_communities()
        
        for community in communities:
            try:
                # Get community subgraph data
                subgraph_data = self._get_community_subgraph(community.nodes)
                
                # Generate summary using LLM
                summary = self._generate_community_summary(
                    subgraph_data, 
                    llm_processor
                )
                
                # Update community summary
                community.summary = summary
                community.last_updated = datetime.now()
                
                # Store summary in Neo4j
                self._store_community_summary(community)
                
            except Exception as e:
                self.logger.error(f"Error updating summary for community {community.id}: {str(e)}")

    def _get_community_subgraph(self, node_ids: Set[str]) -> Dict[str, Any]:
        """Get subgraph data for a community."""
        query = """
        MATCH (n)-[r]->(m)
        WHERE n.id IN $node_ids AND m.id IN $node_ids
        RETURN 
            n.text as source_text,
            type(r) as relationship_type,
            m.text as target_text
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_ids=list(node_ids))
            return result.data()

    def _generate_community_summary(self, subgraph_data: Dict[str, Any], 
                                  llm_processor) -> str:
        """Generate natural language summary of community using LLM."""
        # Format subgraph data for LLM
        prompt = self._format_community_prompt(subgraph_data)
        
        # Get summary from LLM
        summary = llm_processor.generate_text(prompt)
        
        return summary

    def _format_community_prompt(self, subgraph_data: Dict[str, Any]) -> str:
        """Format community data into prompt for LLM."""
        prompt = "Generate a concise summary of the following graph community:\n\n"
        
        # Add relationships
        prompt += "Relationships:\n"
        for rel in subgraph_data:
            prompt += f"- {rel['source_text']} {rel['relationship_type']} {rel['target_text']}\n"
        
        prompt += "\nProvide a natural language summary that describes the key entities and their relationships in this community."
        
        return prompt

    def _store_community_summary(self, community: Community) -> None:
        """Store community summary in Neo4j."""
        query = """
        MERGE (c:Community {id: $id})
        SET 
            c.summary = $summary,
            c.metadata = $metadata,
            c.created_at = datetime($created_at),
            c.last_updated = datetime($last_updated)
        WITH c
        UNWIND $node_ids as node_id
        MATCH (n:Entity {id: node_id})
        MERGE (n)-[:BELONGS_TO]->(c)
        """
        
        with self.neo4j._driver.session() as session:
            session.run(
                query,
                id=community.id,
                summary=community.summary,
                metadata=community.metadata,
                created_at=community.created_at.isoformat(),
                last_updated=community.last_updated.isoformat(),
                node_ids=list(community.nodes)
            )