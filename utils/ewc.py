import torch
import torch.nn as nn
from copy import deepcopy

class EWC:
    def __init__(self, model: nn.Module, ewc_lambda=1.0):
        """
        初始化EWC
        Args:
            model: 需要保护的模型
            ewc_lambda: EWC损失的权重
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # 保存重要参数
        self.saved_parameters = {}
        # 保存Fisher信息
        self.fisher_dict = {}
        
    def update_fisher_parameters(self, train_loader, optimizer):
        """计算Fisher信息和保存重要参数"""
        # 将模型设置为训练模式
        self.model.train()
        
        # 初始化Fisher信息字典
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        # 计算Fisher信息
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            titan_features, uni_features, label, event_time, c = batch
            data_WSI = (titan_features.cuda(), uni_features.cuda())
            data_omic = torch.zeros((1,1)).cuda()
            hazards, S, Y_hat, _, _ = self.model(data_WSI[0], data_WSI[1])
            loss = -torch.sum(S, dim=1).mean()  # 使用负log似然作为损失
            loss.backward()
            
            # 累积Fisher信息
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
        
        # 平均Fisher信息
        for name in fisher_dict:
            fisher_dict[name] /= len(train_loader)
            
        # 保存当前参数
        self.saved_parameters = {name: param.data.clone()
                               for name, param in self.model.named_parameters()}
        self.fisher_dict = fisher_dict
    
    def ewc_loss(self):
        """计算EWC损失"""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and name in self.saved_parameters:
                loss += (self.fisher_dict[name] * (param.data - self.saved_parameters[name]) ** 2).sum()
        return self.ewc_lambda * loss