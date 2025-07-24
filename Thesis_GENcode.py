import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import random
from gensim.models import Word2Vec

# ------------------------------------------
# 1) Spatial Embedding via DeepWalk
# ------------------------------------------
def generate_spatial_embeddings(graph: nx.Graph, emb_dim=128, walks_per_node=10, walk_length=40):
    # Generate random walks
    walks = []
    nodes = list(graph.nodes())
    for _ in range(walks_per_node):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))
            walks.append([str(n) for n in walk])
    
    # Train skip-gram (DeepWalk)
    w2v = Word2Vec(
        sentences=walks,
        vector_size=emb_dim,
        window=5,
        min_count=0,
        sg=1,
        workers=4,
        epochs=5
    )
    
    # Build embedding matrix
    emb_matrix = torch.zeros((graph.number_of_nodes(), emb_dim))
    for node in graph.nodes():
        emb_matrix[node] = torch.tensor(w2v.wv[str(node)])
    return emb_matrix

# Example usage:
# G = nx.grid_2d_graph(50,50)  # or load your spatial graph
# spatial_embs = generate_spatial_embeddings(G, emb_dim=128)

# ------------------------------------------
# 2) Dice Activation Unit (from DIN)
# ------------------------------------------
class Dice(nn.Module):
    def __init__(self, dim=0, eps=1e-9):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # x: [batch, dim]
        normalized = self.bn(x)
        p = self.sigmoid(normalized)
        return p * x + (1 - p) * self.alpha * x

# ------------------------------------------
# 3) RSTP Model Definition
# ------------------------------------------
class RSTPModel(nn.Module):
    def __init__(
        self,
        num_grids: int,
        grid_emb_dim: int,
        attribute_dims: dict,
        attr_emb_dim: int,
        lstm_hidden: int,
        fusion_dim: int
    ):
        super().__init__()
        # Spatial embeddings (learned or precomputed)
        self.grid_embeddings = nn.Embedding(num_grids, grid_emb_dim)
        
        # Attribute embeddings
        self.attr_embeddings = nn.ModuleDict({
            name: nn.Embedding(size, attr_emb_dim)
            for name, size in attribute_dims.items()
        })
        
        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=grid_emb_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # Fusion & interaction
        input_dim = attr_emb_dim * len(attribute_dims) + lstm_hidden + grid_emb_dim
        self.fusion_linear = nn.Linear(input_dim, fusion_dim)
        self.dice = Dice(dim=fusion_dim)
        
        # Output MLP -> maps fused vector to grid_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, grid_emb_dim)
        )
    
    def forward(self, attr_inputs: dict, spatial_seq: torch.LongTensor):
        """
        attr_inputs: dict of {feature_name: LongTensor[batch]}
        spatial_seq: LongTensor[batch, seq_len] of grid IDs
        """
        batch_size = spatial_seq.size(0)
        
        # 1) Attribute encoding
        attr_vecs = []
        for name, emb in self.attr_embeddings.items():
            attr_vecs.append(emb(attr_inputs[name]))
        attr_concat = torch.cat(attr_vecs, dim=1)  # [batch, attr_emb_dim * num_features]
        
        # 2) Temporal encoding via LSTM
        # first embed spatial_seq via grid_embeddings
        seq_emb = self.grid_embeddings(spatial_seq)  # [batch, seq_len, grid_emb_dim]
        _, (h_n, _) = self.lstm(seq_emb)
        temporal_vec = h_n[-1]  # [batch, lstm_hidden]
        
        # 3) Last visited spatial embedding
        last_spatial = spatial_seq[:, -1]  # [batch]
        spatial_vec = self.grid_embeddings(last_spatial)  # [batch, grid_emb_dim]
        
        # 4) Fusion & interaction
        fusion_input = torch.cat([attr_concat, temporal_vec, spatial_vec], dim=1)
        x = self.fusion_linear(fusion_input)
        x = self.dice(x)
        
        # 5) Output vector
        traj_vec = self.mlp(x)  # [batch, grid_emb_dim]
        
        # 6) Score candidates by cosine similarity
        grid_matrix = self.grid_embeddings.weight  # [num_grids, grid_emb_dim]
        # Expand for batched cosine similarity
        traj_exp = traj_vec.unsqueeze(1)  # [batch, 1, grid_emb_dim]
        grid_exp = grid_matrix.unsqueeze(0)  # [1, num_grids, grid_emb_dim]
        scores = F.cosine_similarity(traj_exp, grid_exp, dim=-1)  # [batch, num_grids]
        
        return scores

# ------------------------------------------
# Example instantiation
# ------------------------------------------
# attribute_dims = {
#     'mode': 5,        # e.g., walk, bike, car, bus, train
#     'day_of_week': 7,
#     'time_slot': 24,
#     'user_id': 2000
# }
# model = RSTPModel(
#     num_grids=2500,
#     grid_emb_dim=128,
#     attribute_dims=attribute_dims,
#     attr_emb_dim=32,
#     lstm_hidden=64,
#     fusion_dim=128
# )
# ------------------------------------------
# Training & inference loops would follow,
# using scores = model(attr_inputs, spatial_seq)
# and optimizing a recommendation-style loss.

