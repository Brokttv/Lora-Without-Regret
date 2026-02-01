import torch
import torch.nn as nn
import math


class LoraLayer(nn.Module):
    def __init__(self, rank, alpha, base_map):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.base_map = base_map
        
        self.lora_A = nn.Parameter(torch.empty(rank, base_map.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_map.out_features, rank))
        
        scale = 1 / math.sqrt(base_map.in_features)
        nn.init.uniform_(self.lora_A, -scale, scale)
    
    def forward(self, x):
        term_1 = self.base_map(x)
        term_2 = (x @ self.lora_A.T) @ self.lora_B.T
        return term_1 + (self.alpha / self.rank) * term_2


def inject_lora(model, rank, alpha, device):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            targets.append((name, module))
            
    #Applying Lora to all layers
    for name, module in targets:
        lora_layer = LoraLayer(rank, alpha, module)
        lora_layer.to(device)
        
        name_parts = name.split(".")
        parent_parts = name_parts[:-1]
        child_name = name_parts[-1]
        
        if parent_parts:
            parent_path = ".".join(parent_parts)
            parent_module = model.get_submodule(parent_path)
            setattr(parent_module, child_name, lora_layer)
        else:
            setattr(model, child_name, lora_layer)


def FullFT_or_Lora(model, use_lora, rank, alpha, device):
    if use_lora:
        for param in model.parameters():
            param.requires_grad = False
        inject_lora(model, rank, alpha, device)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        trainable_params = model.parameters()
    
    return trainable_params
