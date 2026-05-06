import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [B, N, C]
        b, n, c = x.size()
        
        # Calculate mean and variance across N (spatial/sequence dimension)
        # We want to enhance features based on their spatial distribution
        # For 1D sequence, dim=1 is N
        x_minus_mu_square = (x - x.mean(dim=1, keepdim=True)).pow(2)
        
        # Energy function
        # y = E_inv = (t - mu)^2 / (4 * (sigma^2 + lambda)) + 0.5
        # sigma^2 = sum(x - mu)^2 / (n-1)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=1, keepdim=True) / (n - 1) + self.e_lambda)) + 0.5
        
        return x * self.activaton(y)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        # qkv: [B, N, 3*C] -> [B, N, 3, H, D] -> [3, B, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attn: [B, H, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x: [B, H, N, D] -> [B, N, H, D] -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AgentAttention(nn.Module):
    """
    Simplified Agent Attention for MIL
    Uses agent tokens to aggregate global context and then distribute it back.
    Complexity: O(N*A) where A is number of agents (small).
    """
    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0., proj_drop=0., agent_num=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.agent_num = agent_num

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Agent tokens: [1, A, D]
        self.agent_tokens = nn.Parameter(torch.randn(1, agent_num, dim))
        self.agent_proj = nn.Linear(dim, dim) # To project agents to query space if needed

    def forward(self, x):
        B, N, C = x.shape
        # qkv: [B, N, 3, H, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, H, N, D]
        
        # Prepare Agents
        # We treat agents as Queries to pool info from x (Keys/Values)
        # agent_tokens: [1, A, C] -> [B, A, C]
        agents = self.agent_tokens.expand(B, -1, -1)
        # Project agents to multiple heads: [B, A, H, D] -> [B, H, A, D]
        q_agent = self.agent_proj(agents).reshape(B, self.agent_num, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 1. Agents aggregate global info
        # q_agent: [B, H, A, D]
        # k: [B, H, N, D]
        # attn_agent: [B, H, A, N]
        attn_agent = (q_agent @ k.transpose(-2, -1)) * self.scale
        attn_agent = attn_agent.softmax(dim=-1)
        attn_agent = self.attn_drop(attn_agent)
        
        # agent_global: [B, H, A, N] @ [B, H, N, D] -> [B, H, A, D]
        agent_global = attn_agent @ v
        
        # 2. Distribute info back to patches
        # q: [B, H, N, D] (Patches query the global agents)
        # k_agent = agent_global (Agents act as keys)
        # v_agent = agent_global (Agents act as values)
        
        # attn_broadcast: [B, H, N, A]
        attn_broadcast = (q @ agent_global.transpose(-2, -1)) * self.scale
        attn_broadcast = attn_broadcast.softmax(dim=-1)
        attn_broadcast = self.attn_drop(attn_broadcast)
        
        # x_out: [B, H, N, A] @ [B, H, A, D] -> [B, H, N, D]
        x_out = attn_broadcast @ agent_global
        
        # Reshape and project
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        # Residual connection is usually handled outside or we can add it here if x shape matches
        return x + x_out