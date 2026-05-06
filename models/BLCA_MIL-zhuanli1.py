# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
# https://arxiv.org/pdf/1802.04712.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
try:
    from models.custom_modules import SimAM, AgentAttention, ScaledDotProductAttention
except ImportError:
    # Fallback if custom_modules not found (e.g. not created yet)
    pass

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
        
        # New modules
        self.simam = SimAM()
        self.agent_attn = AgentAttention(dim=L, num_heads=8, agent_num=8)
        self.use_advanced_modules = True # Flag to enable/disable
        
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
        
        # Apply SimAM to enhance features (Parameter-free)
        if self.use_advanced_modules:
            x = self.simam(x)
            
        feature = self.feature(x)
        # feature = feature.squeeze() # Remove squeeze to keep sequence dim [B, N, L]
        
        # Apply Agent Attention for global context modeling
        if self.use_advanced_modules:
            feature = self.agent_attn(feature)
        
        feature_seq = feature # Keep sequence for attention
        
        # Collapse for MIL attention (if needed) or use sequence
        # The original code expects feature to be [N, L] (squeezed B=1) or [B, N, L]
        # But original code had feature.squeeze() which implies B=1 usually in this legacy code?
        # Let's check if B is always 1.
        # "x1 = x1.unsqueeze(1)" -> [B, 1, 768]
        # "x2 = x2.unsqueeze(0).expand(B, -1, -1)"
        # So B is preserved.
        # Original: feature = feature.squeeze()
        # If B=1, [1, N, L] -> [N, L].
        # If B>1, [B, N, L] -> [B, N, L] (squeeze doesn't remove dim if size != 1)
        # However, if B=1, squeeze() removes the first dim.
        # Let's handle B explicitly.
        
        if feature.size(0) == 1:
            feature_flat = feature.squeeze(0) # [N, L]
        else:
            feature_flat = feature # [B, N, L] - Attention expects this?
            # Original code:
            # A = self.attention(feature)
            # A = torch.transpose(A, -1, -2) # KxN
            # M = torch.mm(A, feature) # KxL
            # This implies feature is 2D [N, L] and A is [N, K].
            # So the original code likely supports B=1 only.
            
        # We need to maintain compatibility with B=1 constraint of original code or adapt it.
        # Assuming B=1 for safety as per original "squeeze()".
        
        if len(feature.shape) == 3 and feature.shape[0] == 1:
             feature = feature.squeeze(0)
             
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





if __name__ == '__main__':
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    model = DAttention(1024, 2, dropout=False, act='relu')

