# database/q_learning.py
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime
import random
from collections import deque
import json

from ..config import Config
from .graph import Neo4jConnection

@dataclass
class Experience:
    """Single training experience for Q-learning."""
    state: str
    action: str
    reward: float
    next_state: str
    done: bool

class ExperienceBuffer:
    """Buffer for storing and sampling experiences."""
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self) -> int:
        return len(self.buffer)

class QNetwork(nn.Module):
    """Neural network for Q-value prediction."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
                
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class QValueTrainer:
    """Handles Q-value network training and optimization."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Training parameters
        self.batch_size = config.model_config.get('batch_size', 32)
        self.learning_rate = config.model_config.get('learning_rate', 1e-4)
        self.gamma = config.model_config.get('gamma', 0.99)
        self.tau = config.model_config.get('tau', 0.005)
        
        # Initialize networks
        self.state_dim = config.model_config.get('embedding_dim', 768)
        self.action_dim = config.model_config.get('embedding_dim', 768)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #adjust for M3 Chip for macbook
        
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.experience_buffer = ExperienceBuffer()
        
        # Training metrics
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'q_values': []
        }

    def train(self, embedding_model, num_episodes: int = 1000) -> Dict[str, List[float]]:
        """Train Q-network using experience replay."""
        self.logger.info(f"Starting Q-network training for {num_episodes} episodes")
        
        try:
            for episode in range(num_episodes):
                # Generate training experience
                experiences = self._generate_episode_experience(embedding_model)
                for exp in experiences:
                    self.experience_buffer.add(exp)
                
                # Train on mini-batches
                if len(self.experience_buffer) >= self.batch_size:
                    loss = self._train_step(embedding_model)
                    self.training_metrics['losses'].append(loss)
                
                # Update target network
                if episode % 100 == 0:
                    self._update_target_network()
                    
                if episode % 10 == 0:
                    self.logger.info(f"Episode {episode}, Loss: {loss:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"Error during Q-network training: {str(e)}")
            raise

    def predict_q_value(self, state_embedding: torch.Tensor, 
                       action_embedding: torch.Tensor) -> float:
        """Predict Q-value for state-action pair."""
        with torch.no_grad():
            self.q_network.eval()
            q_value = self.q_network(
                state_embedding.to(self.device),
                action_embedding.to(self.device)
            )
            return q_value.item()

    def _generate_episode_experience(self, 
                                   embedding_model) -> List[Experience]:
        """Generate experience from random walks in the graph."""
        experiences = []
        
        try:
            # Get random starting node
            start_node = self._get_random_node()
            current_state = start_node
            
            max_steps = 20
            for step in range(max_steps):
                # Get possible actions (neighboring nodes)
                actions = self._get_possible_actions(current_state)
                if not actions:
                    break
                    
                # Select random action for exploration
                action = random.choice(actions)
                
                # Get next state
                next_state = self._take_action(current_state, action)
                
                # Calculate reward
                reward = self._calculate_reward(current_state, action, next_state)
                
                # Store experience
                experience = Experience(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(step == max_steps - 1)
                )
                
                experiences.append(experience)
                current_state = next_state
                
            return experiences
            
        except Exception as e:
            self.logger.error(f"Error generating experience: {str(e)}")
            return []

    def _train_step(self, embedding_model) -> float:
        """Perform single training step."""
        self.q_network.train()
        
        try:
            # Sample batch of experiences
            experiences = self.experience_buffer.sample(self.batch_size)
            
            # Prepare batch
            state_embeddings = []
            action_embeddings = []
            rewards = []
            next_state_embeddings = []
            dones = []
            
            for exp in experiences:
                # Get embeddings
                state_emb = self._get_node_embedding(exp.state, embedding_model)
                action_emb = self._get_node_embedding(exp.action, embedding_model)
                next_state_emb = self._get_node_embedding(exp.next_state, embedding_model)
                
                state_embeddings.append(state_emb)
                action_embeddings.append(action_emb)
                rewards.append(exp.reward)
                next_state_embeddings.append(next_state_emb)
                dones.append(float(exp.done))
            
            # Convert to tensors
            states = torch.stack(state_embeddings).to(self.device)
            actions = torch.stack(action_embeddings).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.stack(next_state_embeddings).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
            
            # Calculate current Q values
            current_q = self.q_network(states, actions)
            
            # Calculate target Q values
            with torch.no_grad():
                # Get Q values for next states
                next_q = self.target_network(next_states, actions)
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            # Calculate loss and update
            loss = F.mse_loss(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            return float('inf')

    def _update_target_network(self):
        """Update target network using soft update."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _get_random_node(self) -> str:
        """Get random node from graph."""
        query = """
        MATCH (n:Entity)
        RETURN n.id
        ORDER BY rand()
        LIMIT 1
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record[0] if record else None

    def _get_possible_actions(self, node_id: str) -> List[str]:
        """Get possible actions (neighboring nodes) for a state."""
        query = """
        MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)
        RETURN m.id
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_id=node_id)
            return [record[0] for record in result]

    def _take_action(self, state: str, action: str) -> str:
        """Take action and get next state."""
        query = """
        MATCH (n:Entity {id: $state_id})-[r]-(m:Entity {id: $action_id})
        RETURN m.id
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, state_id=state, action_id=action)
            record = result.single()
            return record[0] if record else None

    def _calculate_reward(self, state: str, action: str, next_state: str) -> float:
        """Calculate reward for state-action-next_state transition."""
        try:
            # Get node properties
            query = """
            MATCH (s:Entity {id: $state_id})
            MATCH (a:Entity {id: $action_id})
            MATCH (n:Entity {id: $next_id})
            RETURN s.type as s_type, a.type as a_type, n.type as n_type,
                   exists((s)-[]-(a)) as s_a_connected,
                   exists((a)-[]-(n)) as a_n_connected
            """
            
            with self.neo4j._driver.session() as session:
                result = session.run(
                    query,
                    state_id=state,
                    action_id=action,
                    next_id=next_state
                )
                record = result.single()
                
                if not record:
                    return 0.0
                
                # Base reward for valid transition
                reward = 1.0
                
                # Bonus for maintaining connectivity
                if record['s_a_connected'] and record['a_n_connected']:
                    reward += 0.5
                
                # Bonus for type transitions
                if record['s_type'] != record['n_type']:
                    reward += 0.3  # Encourage exploring different types
                
                return reward
                
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0

    def save_model(self, path: str):
        """Save Q-network model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_metrics': self.training_metrics
        }, path)

    def load_model(self, path: str):
        """Load Q-network model."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_metrics = checkpoint['training_metrics']
        self.q_network.eval()
        self.target_network.eval()