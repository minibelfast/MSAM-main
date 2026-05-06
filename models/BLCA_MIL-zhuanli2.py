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






# --- AAAI 2025 SNN & Attention Modules ---

class SurrogateHeaviside(torch.autograd.Function):
    """
    Surrogate Gradient for Spiking Neurons (Heaviside step function with custom backward).
    Inspired by: CREST & Adaptive Calibration papers.
    """
    @staticmethod
    def forward(ctx, input, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Sigmoid derivative approximation for gradient
        sigmoid_grad = (1 / ctx.alpha) * (torch.sigmoid(ctx.alpha * input) * (1 - torch.sigmoid(ctx.alpha * input)))
        return grad_output * sigmoid_grad, None

class LIFNode(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron.
    Simulates temporal dynamics or acts as a spiking activation.
    Inspired by: Spiking Point Transformer & Noise-Injected SNN papers.
    """
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, alpha=2.0):
        super(LIFNode, self).__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = SurrogateHeaviside.apply
        self.alpha = alpha
        self.membrane_potential = 0.0

    def forward(self, x):
        # x: [Batch, ..., Dim]
        # Handle state initialization/reset
        if isinstance(self.membrane_potential, float) or isinstance(self.membrane_potential, int):
             self.membrane_potential = torch.zeros_like(x)
        elif self.membrane_potential.shape != x.shape:
             self.membrane_potential = torch.zeros_like(x)
             
        # Integrate
        # Simple Euler method for LIF: V[t] = V[t-1] + (Input - V[t-1])/tau
        self.membrane_potential = self.membrane_potential.to(x.device) + (x - self.membrane_potential.to(x.device)) / self.tau
        
        # Fire
        spike = self.surrogate_function(self.membrane_potential - self.v_threshold, self.alpha)
        
        # Soft Reset: V = V - V_th * spike
        self.membrane_potential = self.membrane_potential - (self.v_threshold * spike)
        
        return spike

class FrequencyEnhance(nn.Module):
    """
    Frequency-based Spatial-Temporal Attention (FSTA) Module.
    Uses FFT to enhance features in the frequency domain, reducing redundancy.
    Inspired by: FSTA-SNN paper.
    """
    def __init__(self, channels):
        super(FrequencyEnhance, self).__init__()
        self.channels = channels
        # Learnable frequency gate
        # FFT of real signal of length C results in C//2 + 1 complex components
        self.freq_gate = nn.Parameter(torch.ones(channels // 2 + 1, dtype=torch.float32))

    def forward(self, x):
        # x: [B, N, C]
        # Apply FFT along the last dimension (Feature dimension)
        x_fft = torch.fft.rfft(x, dim=-1) # [B, N, C//2 + 1]
        
        # Apply gate (broadcasting along B and N)
        # Convert gate to complex for scaling amplitude
        gate = torch.sigmoid(self.freq_gate).unsqueeze(0).unsqueeze(0)
        
        # Element-wise multiplication
        x_enhanced_fft = x_fft * gate
        
        # Inverse FFT
        x_enhanced = torch.fft.irfft(x_enhanced_fft, n=x.shape[-1], dim=-1)
        
        # Residual connection
        return x + x_enhanced

class TSE_Attention(nn.Module):
    """
    Temporal-Self-Erasing (TSE) Attention.
    Performs multi-step attention, masking out high-attention regions from previous steps.
    Inspired by: Towards More Discriminative Feature Learning in SNNs.
    """
    def __init__(self, L=512, D=128, K=1, steps=2, drop_ratio=0.1):
        super(TSE_Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.steps = steps
        self.drop_ratio = drop_ratio 
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x):
        # x: [B, N, L]
        B, N, C = x.shape
        
        all_features = []
        all_attention = []
        
        # Mask for valid instances (initially all 1s, 0 means masked)
        # Shape: [B, N, 1]
        mask = torch.ones(B, N, 1, device=x.device) 
        
        for s in range(self.steps):
            # Compute Attention Scores (logits)
            A_logits = self.attention(x) # [B, N, K]
            
            # Apply mask: Set logits of masked instances to -inf
            # masked_fill expects boolean mask where True values are filled
            A_logits = A_logits.masked_fill(mask == 0, -1e9)
            
            # Softmax
            A_scores = F.softmax(A_logits, dim=1) # [B, N, K]
            
            # Aggregate: M = A^T * X
            M = torch.bmm(A_scores.transpose(1, 2), x) # [B, K, L]
            
            all_features.append(M)
            all_attention.append(A_scores)
            
            if s < self.steps - 1:
                # Update Mask: Drop top-k high attention instances for next step
                k_drop = max(1, int(N * self.drop_ratio))
                
                # Get indices of top-k attention scores
                # We use sum over K if K > 1, but usually K=1 in MIL
                scores_flat = A_scores.sum(dim=2) # [B, N]
                _, indices = torch.topk(scores_flat, k_drop, dim=1) # [B, k_drop]
                
                # Update mask
                # indices gives the positions to mask out
                # Create a scatter mask
                mask_update = torch.zeros_like(mask).scatter_(1, indices.unsqueeze(-1), 1.0)
                mask = mask * (1 - mask_update) # Set masked positions to 0
                    
        # Combine features from all steps (Average)
        final_feature = torch.stack(all_features, dim=1).mean(dim=1).squeeze(1) # [B, L]
        
        return final_feature, all_attention

class SpikingMIL(nn.Module):
    """
    Spiking MIL Model for Bladder Cancer Prognosis.
    Integrates SNN dynamics, Frequency Enhancement, and Self-Erasing Attention.
    """
    def __init__(self, in_dim_titan=768, in_dim_uni=1536, n_classes=2, dropout=True, act='snn', survival=False):
        super(SpikingMIL, self).__init__()
        self.L = 512
        self.survival = survival
        
        # 1. Projections with LIF or GELU
        self.titan_proj = nn.Sequential(
            nn.Linear(in_dim_titan, self.L),
            nn.LayerNorm(self.L),
            LIFNode(tau=2.0) if act == 'snn' else nn.GELU(),
            nn.Dropout(0.25 if dropout else 0)
        )
        
        self.uni_proj = nn.Sequential(
            nn.Linear(in_dim_uni, self.L),
            nn.LayerNorm(self.L),
            LIFNode(tau=2.0) if act == 'snn' else nn.GELU(),
            nn.Dropout(0.25 if dropout else 0)
        )
        
        # 2. Frequency Enhancement
        self.freq_enhance = FrequencyEnhance(self.L)
        
        # 3. TSE Attention
        self.tse_attention = TSE_Attention(L=self.L, steps=2, drop_ratio=0.1)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.L, n_classes)
        )
        
        # Initialize LIF state
        self.lif_nodes = [m for m in self.modules() if isinstance(m, LIFNode)]

    def forward(self, x1, x2):
        # Reset LIF states at the beginning of each forward pass (for stateless batch processing)
        for node in self.lif_nodes:
            node.membrane_potential = 0.0

        # x1: Titan [B, 768] -> [B, 1, 768]
        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
        
        # x2: Uni [B, N, 1536]
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0) # [1, N, 1536]
            
        # Process Titan features
        x1_feat = self.titan_proj(x1) # [B, 1, L]
        
        # Process Uni features
        x2_feat = self.uni_proj(x2) # [B, N, L]
        
        # Concatenate features: [B, N+1, L]
        # Assuming we want to attend to both Titan (slide-level?) and Uni (patch-level?)
        x = torch.cat([x1_feat, x2_feat], dim=1) 
        
        # Apply Frequency Enhancement
        x = self.freq_enhance(x)
        
        # Apply TSE Attention
        feature, attentions = self.tse_attention(x)
        
        # Classifier
        logits = self.classifier(feature)
        
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
            
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        # Return first step attention for visualization compatibility
        return logits, Y_prob, Y_hat, attentions[0], {}

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)


if __name__ == '__main__':

    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    model = DAttention(1024, 2, dropout=False, act='relu')

