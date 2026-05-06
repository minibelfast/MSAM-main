import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MeanMIL(nn.Module):
    def __init__(self, in_dim=1024, n_classes=1, dropout=True, act='relu', survival = False, L=512, in_dim_titan=768, in_dim_uni=1536):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(L,L)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)
        self.survival = survival
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
        if len(x.shape) == 3 and x.shape[0] > 1:
            raise RuntimeError('Batch size must be 1, current batch size is:{}'.format(x.shape[0]))
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x[0]
        logits = self.head(x)
        logits = torch.mean(logits, dim=0, keepdim=True)
        
        '''
        Survival Layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(device)



class MaxMIL(nn.Module):
    def __init__(self, in_dim=1024, n_classes=1, dropout=True,act='relu', survival = False, L=512, in_dim_titan=768, in_dim_uni=1536):
        super(MaxMIL, self).__init__()

        head = [nn.Linear(L,L)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)
        
        self.survival = survival

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
        if len(x.shape) == 3 and x.shape[0] > 1:
            raise RuntimeError('Batch size must be 1, current batch size is:{}'.format(x.shape[0]))
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x[0]
        
        logits = self.head(x)
        logits, _ = torch.max(logits, dim=0, keepdim=True)
        
        '''
        Survival Layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        return logits, Y_prob, Y_hat, A_raw, results_dict        
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(device)


if __name__ == '__main__':
    mean_model = MeanMIL(n_classes=2)
    x = torch.randn(100, 1024)
    y = mean_model(x)
    print(y)