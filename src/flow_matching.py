import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding for the timestep 't'.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowMatchingActionExpert(nn.Module):
    """
    The 'Action Expert' network v_theta.
    Predicts the vector field (velocity) to transport noise -> expert trajectory.
    Ref: Alpamayo-R1, Section 5.1
    """
    def __init__(self, 
                 state_dim=4,       # [x, y, v, yaw]
                 hidden_dim=256,    # Lightweight dimension for speed
                 context_dim=256,   # Dimension of VectorNet output
                 n_heads=4,
                 n_layers=3,        # Shallow depth for <10ms inference
                 dropout=0.1):
        super().__init__()

        # 1. Embeddings
        self.traj_proj = nn.Linear(state_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),  # Keep SiLU here in the MLP, it is valid as a module
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 2. Transformer Decoder Blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"  # <--- FIXED: Changed from "silu" to "gelu"
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 3. Output Head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, noisy_traj, timestep, context_emb):
        """
        Args:
            noisy_traj: [Batch, Horizon, State_Dim]
            timestep:   [Batch] (Scalar time t in [0, 1])
            context_emb:[Batch, Context_Len, Context_Dim]
        """
        B, H, _ = noisy_traj.shape
        
        # A. Project Trajectory to Latent Space
        x_h = self.traj_proj(noisy_traj)

        # B. Inject Time Embedding
        t_emb = self.time_mlp(timestep).unsqueeze(1)
        x_h = x_h + t_emb

        # C. Cross-Attend to Context
        x_h = self.transformer(tgt=x_h, memory=context_emb)

        # D. Predict Vector Field (Velocity)
        v_pred = self.output_head(x_h)

        return v_pred