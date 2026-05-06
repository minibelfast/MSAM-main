# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
# https://arxiv.org/pdf/1802.04712.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class DAttention(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, L=512, in_dim_titan=768, in_dim_uni=1536):
        super(DAttention, self).__init__()
        self.L = L
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(L, L)]
        self.survival = survival
        
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )


        # self.apply(initialize_weights)


        # Titan特征处理
        self.titan_proj = nn.Sequential(
            nn.Linear(in_dim_titan, L),
            nn.LayerNorm(L),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Uni特征处理
        self.uni_proj = nn.Sequential(
            nn.Linear(in_dim_uni, L),
            nn.LayerNorm(L),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x1, x2):
        # 处理Titan特征
        x1 = x1.unsqueeze(1)  # 添加patch维度 [B, 1, 768]
        x1 = self.titan_proj(x1)  # [B, 1, L]
        
        # 处理Uni特征
        B = x1.size(0)
        x2 = x2.unsqueeze(0).expand(B, -1, -1)  # [B, n_patches, 1536]
        x2 = self.uni_proj(x2)  # [B, n_patches, L]
        
        # 特征融合
        x = torch.cat([x1, x2], dim=1)  # [B, n_patches+1, L]
        feature = self.feature(x)
        feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A_raw = A
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        
        logits = self.classifier(M)
        
        '''
        Survival layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None 
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # keep the same API with the clam
        return logits, Y_prob, Y_hat, A_raw, {}
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature = self.feature.to(device)
        self.attention = self.attention.to(device)
        self.classifier = self.classifier.to(device)



class GatedAttention(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(in_dim, 512)]
        self.survival = survival
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze()

        A_V = self.attention_V(feature)  # NxD
        A_U = self.attention_U(feature)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, feature)  # KxL

        logits = self.classifier(M)
        
        '''
        Survival layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None 
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # keep the same API with the clam
        return logits, Y_prob, Y_hat, None, {}






# ==================================================================================================
# ICLR 2025 SNN Integration Modules
# ==================================================================================================

class SurrogateHeaviside(torch.autograd.Function):
    """
    Standard surrogate gradient for spiking neurons.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Approximate derivative of Heaviside (e.g., triangular or rectangular)
        grad_input = grad_output.clone()
        # Rectangular window: gradient is 1 where |input| < 0.5 (or similar width)
        grad_input[torch.abs(input) > 0.5] = 0
        return grad_input

class IELIFNode(nn.Module):
    """
    Implements concepts from:
    1. Paper 1 (Ensemble): Membrane Potential Smoothing.
    2. Paper 2 (Quantized Spike-driven Transformer): Information-Enhanced LIF (IE-LIF).
       - Note: Full IE-LIF uses multi-bit spikes. Here we implement a simplified version 
         with learnable smoothing and potential correction.
    """
    def __init__(self, tau=2.0, decay_input=False, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.tau = tau
        self.decay_input = decay_input
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        # Paper 1: Membrane Potential Smoothing Coefficient (Learnable)
        # "using a learnable smoothing coefficient to smooth the pre-charging membrane potential"
        self.smooth_coef = nn.Parameter(torch.tensor(0.1))
        
        # Paper 2: IE-LIF Potential Correction (Learnable Shift/Scale to make it Gaussian-like)
        self.norm = nn.LayerNorm(1) # Applied per neuron effectively if shape matches

    def forward(self, x, v_prev=None):
        # x: [Batch, ..., Dim]
        if v_prev is None:
            v_prev = torch.zeros_like(x)
            
        # Neuronal Dynamics
        # V[t] = V[t-1] * (1 - 1/tau) + X[t] (simplified LIF)
        decay = 1.0 / self.tau
        
        # Paper 1: Membrane Potential Smoothing
        # Instead of raw update, we smooth the integration
        # V_pre = (1 - alpha) * (Decay * V_prev + X) + alpha * V_smooth_prev ? 
        # Simplified interpretation: Smooth the input current or the potential itself.
        # Let's implement a residual smoothing on the potential update.
        
        v_integrated = (1 - decay) * v_prev + x
        
        # Apply smoothing: v_smooth = v_integrated + sigmoid(coef) * v_prev (guidance)
        # Or strictly following paper: "smooth the pre-charging membrane potential"
        # We'll use a weighted average with the previous potential to stabilize.
        alpha = torch.sigmoid(self.smooth_coef)
        v_membrane = (1 - alpha) * v_integrated + alpha * v_prev
        
        # Spike Generation
        spike = SurrogateHeaviside.apply(v_membrane - self.v_threshold)
        
        # Reset
        v_next = v_membrane - spike * (self.v_threshold - self.v_reset)
        
        return spike, v_next

class WeightRescaler(nn.Module):
    """
    Paper 4: QP-SNN - Weight Rescaling Strategy.
    Scales weights to utilize the full dynamic range for quantization (or representation).
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Scale weights during forward pass (simulated)
        # In a real quantized setting, we would quantize here.
        # For floating point SNN, this acts as a dynamic gain control.
        return self.module(x) * self.scale

class SpikingSelfAttention(nn.Module):
    """
    Paper 6: Spiking Vision Transformer (Saccadic Attention concepts).
    Paper 2: Quantized Spike-driven Transformer.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Paper 4: Weight Rescaling applied to projections
        self.q_proj = WeightRescaler(nn.Linear(dim, dim))
        self.k_proj = WeightRescaler(nn.Linear(dim, dim))
        self.v_proj = WeightRescaler(nn.Linear(dim, dim))
        self.out_proj = WeightRescaler(nn.Linear(dim, dim))

        # LIF Nodes for Q, K, V (Spike generation)
        self.lif_q = IELIFNode()
        self.lif_k = IELIFNode()
        self.lif_v = IELIFNode()
        
        # LIF Node for Attention Output
        self.lif_out = IELIFNode()

    def forward(self, x, v_states=None):
        # x: [Batch, SeqLen, Dim] (Input at current timestep)
        # v_states: Dictionary storing membrane potentials for Q, K, V, Out nodes
        
        if v_states is None:
            v_states = {
                'q': None, 'k': None, 'v': None, 'out': None
            }

        B, N, C = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Generate Spikes (SNN-ViT logic)
        # Reshape for multi-head: [B, N, H, D]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply LIF (Need to handle shapes for LIF state)
        # We flatten heads for LIF state management or keep them
        # Simplified: Apply LIF on the projected tensors directly
        
        spike_q, v_states['q'] = self.lif_q(q, v_states['q'])
        spike_k, v_states['k'] = self.lif_k(k, v_states['k'])
        spike_v, v_states['v'] = self.lif_v(v, v_states['v'])
        
        # 3. Spiking Attention Mechanism
        # Paper 2/6: Spike-driven attention.
        # A = Q * K^T. Since Q, K are binary (0/1), this is efficient accumulation.
        attn = (spike_q @ spike_k.transpose(-2, -1)) * self.scale
        # In SNNs, Softmax is often replaced or kept. Paper 6 uses Saccadic concepts, 
        # but for compatibility, we use Softmax or a simple normalization.
        attn = attn.softmax(dim=-1)
        
        # 4. Aggregation
        # Output = Attn * V (V is spikes)
        x_out = (attn @ spike_v)
        
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.out_proj(x_out)
        
        # Output Spikes
        spike_out, v_states['out'] = self.lif_out(x_out, v_states['out'])
        
        return spike_out, v_states

class SpikingTransformerMIL(nn.Module):
    """
    Stitched Model integrating:
    - Spiking Transformer Backbone (Paper 2, 6)
    - Ensemble / DeepTAGE Training (Paper 1, 3)
    - Weight Rescaling (Paper 4)
    """
    def __init__(self, in_dim_titan=768, in_dim_uni=1536, L=512, n_classes=2, time_steps=4):
        super().__init__()
        self.L = L
        self.time_steps = time_steps
        self.n_classes = n_classes
        
        # Input Encoders (Titan & Uni)
        self.titan_proj = nn.Linear(in_dim_titan, L)
        self.uni_proj = nn.Linear(in_dim_uni, L)
        
        # Input LIF Node (Paper 1: Membrane Smoothing)
        self.lif_input = IELIFNode()
        
        # Spiking Transformer Encoder (Paper 6)
        self.attn_block = SpikingSelfAttention(L)
        self.norm1 = nn.LayerNorm(L)
        self.mlp = nn.Sequential(
            nn.Linear(L, L * 4),
            nn.GELU(), # Or Spiking MLP
            nn.Linear(L * 4, L)
        )
        self.lif_mlp = IELIFNode()
        
        # Classification Head (Paper 3: DeepTAGE - Auxiliary Classifiers)
        # We share the head across time steps, but compute loss at each step.
        self.classifier = nn.Linear(L, n_classes)

    def forward(self, x1, x2):
        # x1: Titan [B, 768] -> [B, 1, 768]
        # x2: Uni [B, N, 1536]
        
        B = x1.size(0)
        
        # 1. Feature Fusion & Projection
        x1 = x1.unsqueeze(1)
        feat1 = self.titan_proj(x1) # [B, 1, L]
        feat2 = self.uni_proj(x2)   # [B, N, L]
        x = torch.cat([feat1, feat2], dim=1) # [B, N+1, L]
        
        # 2. Spiking Loop over Time Steps (T)
        outputs_per_step = []
        
        # Initialize Membrane Potentials
        v_input = None
        v_attn = None
        v_mlp = None
        
        for t in range(self.time_steps):
            # Input Encoding (Repeat input or Poisson - here Direct Encoding)
            # Paper 1: Treat each timestep as ensemble member
            spike_in, v_input = self.lif_input(x, v_input)
            
            # Transformer Block
            # Residual connection is typical in ViT, but in SNN often add potentials.
            # Here we do simplified Residual on Spikes (X + Attn(X))
            spike_attn, v_attn = self.attn_block(spike_in, v_attn)
            
            # Add & Norm (simplified for SNN)
            x_mid = spike_in + spike_attn
            x_mid = self.norm1(x_mid)
            
            # MLP Block
            x_mlp = self.mlp(x_mid)
            spike_out, v_mlp = self.lif_mlp(x_mlp, v_mlp)
            
            # Global Average Pooling (MIL Aggregation)
            # Aggregate over patches (dim 1)
            # [B, N+1, L] -> [B, L]
            feat_agg = spike_out.mean(dim=1) 
            
            # Classification
            logits = self.classifier(feat_agg)
            outputs_per_step.append(logits)
            
        # Return all time steps for DeepTAGE (Paper 3) and Guidance (Paper 1)
        # Stack: [T, B, n_classes]
        return torch.stack(outputs_per_step, dim=0)

# Loss Function Helper for DeepTAGE and Guidance
def snn_optimization_loss(outputs_per_step, targets, criterion=nn.CrossEntropyLoss()):
    """
    Paper 3: DeepTAGE - Compute loss at each time step (Auxiliary Classifiers).
    Paper 1: Temporally Adjacent Subnetwork Guidance - Distill output[t] from output[t-1].
    """
    T, B, C = outputs_per_step.shape
    total_loss = 0.0
    
    # 1. DeepTAGE: Scale gradients or just sum losses. 
    # Paper 3 suggests scaling based on deviation, but sum is the baseline.
    for t in range(T):
        loss_t = criterion(outputs_per_step[t], targets)
        # Apply time-dependent weight (optional, e.g., increasing weight for later steps)
        weight = (t + 1) / T
        total_loss += weight * loss_t
        
    # 2. Paper 1: Temporally Adjacent Subnetwork Guidance (Consistency)
    # Distill t with t-1
    consistency_loss = 0.0
    if T > 1:
        for t in range(1, T):
            # MSE or KL Div between logits of adjacent steps
            prev_logits = outputs_per_step[t-1].detach() # Stop gradient for teacher
            curr_logits = outputs_per_step[t]
            consistency_loss += F.mse_loss(curr_logits, prev_logits)
            
    return total_loss + 0.1 * consistency_loss

if __name__ == '__main__':
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # model = DAttention(1024, 2, dropout=False, act='relu')
    
    # Test New Spiking Model
    print("Testing SpikingTransformerMIL...")
    model = SpikingTransformerMIL(in_dim_titan=768, in_dim_uni=1536, L=512, n_classes=2, time_steps=4)
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Dummy Input
    B = 2
    N = 100 # patches
    x1 = torch.randn(B, 768)
    x2 = torch.randn(B, N, 1536)
    
    if torch.cuda.is_available():
        x1 = x1.cuda()
        x2 = x2.cuda()
        
    outputs = model(x1, x2)
    print(f"Output Shape: {outputs.shape} (TimeSteps, Batch, Classes)")
    
    # Test Loss
    targets = torch.randint(0, 2, (B,))
    if torch.cuda.is_available():
        targets = targets.cuda()
        
    loss = snn_optimization_loss(outputs, targets)
    print(f"Loss: {loss.item()}")


