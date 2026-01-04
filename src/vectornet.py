import torch
import torch.nn as nn
import torch.nn.functional as F

class PolylineSubGraph(nn.Module):
    """
    The 'Local' layer. 
    Applies a PointNet-like MLP to every point in a polyline, then aggregates 
    them via MaxPool to get a single embedding per polyline.
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, polylines, mask):
        """
        Args:
            polylines: [Batch, Max_Num_Polylines, Max_Points_Per_Poly, In_Channels]
            mask:      [Batch, Max_Num_Polylines, Max_Points_Per_Poly] (1=Valid, 0=Pad)
        Returns:
            polyline_embeddings: [Batch, Max_Num_Polylines, Hidden_Dim]
        """
        B, N, P, C = polylines.shape
        
        # 1. Apply MLP to every point independently
        # x: [B, N, P, Hidden_Dim]
        x = self.mlp(polylines)
        
        # 2. MaxPool over points within a polyline (Symmetric Function)
        # We must mask out padding so max() doesn't pick zeros/garbage
        mask_expanded = mask.unsqueeze(-1) # [B, N, P, 1]
        
        # Set padded values to -inf so they don't affect max
        x = x.masked_fill(~mask_expanded.bool(), float('-1e9'))
        
        # Aggregate: [B, N, Hidden_Dim]
        polyline_embeddings = torch.max(x, dim=2)[0]
        
        return polyline_embeddings

class VectorNetEncoder(nn.Module):
    """
    The 'Global' Context Encoder.
    Integrates Map (Lanes) and Agents (History) into a unified Context Z.
    """
    def __init__(self, 
                 state_dim=6,       # [x, y, z, dir_x, dir_y, type_encoding]
                 hidden_dim=256,    # Matches Action Expert's context_dim
                 n_layers=3,        # Transformer depth
                 n_heads=4):
        super().__init__()
        
        # 1. Local Subgraph (Feature Extraction)
        self.polyline_encoder = PolylineSubGraph(state_dim, hidden_dim)
        
        # 2. Global Graph (Interaction)
        # We use a standard Transformer Encoder to model interactions between
        # map elements and agents.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="relu"
        )
        self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final projection to ensure clean latent space
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, map_polylines, map_mask, agent_polylines, agent_mask):
        """
        Args:
            map_polylines:   [B, N_map, P_map, Dim]
            map_mask:        [B, N_map, P_map]
            agent_polylines: [B, N_agent, P_agent, Dim]
            agent_mask:      [B, N_agent, P_agent]
            
        Returns:
            context_z: [B, N_map + N_agent, Hidden_Dim] 
                       (This is the 'context_emb' for Flow Matching)
        """
        # A. Encode Map and Agents separately via SubGraph
        # map_emb: [B, N_map, Hidden]
        map_emb = self.polyline_encoder(map_polylines, map_mask)
        
        # agent_emb: [B, N_agent, Hidden]
        agent_emb = self.polyline_encoder(agent_polylines, agent_mask)
        
        # B. Concatenate Tokens
        # context_tokens: [B, N_total, Hidden]
        context_tokens = torch.cat([map_emb, agent_emb], dim=1)
        
        # Create a Global Attention Mask (Mask out entirely empty polylines)
        # If a polyline had NO valid points, its max-pool result is garbage (-inf).
        # global_mask: [B, N_total] (True = Ignore/Pad)
        # We define a polyline as invalid if all its points were masked.
        map_valid = map_mask.any(dim=2)     # [B, N_map]
        agent_valid = agent_mask.any(dim=2) # [B, N_agent]
        global_mask = ~torch.cat([map_valid, agent_valid], dim=1) # PyTorch expects True for PAD
        
        # C. Global Interaction (Self-Attention)
        # z: [B, N_total, Hidden]
        z = self.global_transformer(
            src=context_tokens, 
            src_key_padding_mask=global_mask
        )
        
        return self.proj(z)