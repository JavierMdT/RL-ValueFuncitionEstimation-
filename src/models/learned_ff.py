from typing import cast
import os 
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

from src.models import mlp, simba, simbav2
from src.utils import model_hyperparameters

import torch
import torch.nn as nn
import torch.nn.init as init

class ff_embedding(nn.Module):
    """
    Computes Fourier feature embedding: [sin(2*pi*B*x) || cos(2*pi*B*x) || x]
    """
    def __init__(self, in_features: int, out_features: int, sigma: float = 0.1):
        super().__init__()
        
        # Bij <- N(0, sigma^2)
        initial_B = torch.randn(out_features // 2, in_features) * sigma
        
        # B its a learnable parameters
        self.B = nn.Parameter(initial_B)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project inputs: x @ B.T -> shape: (batch_size, out_features // 2)
        proj = 2 * torch.pi * torch.matmul(x, self.B.T)
        
        # Concatenate sin, cos, and the original input x
        return torch.cat([torch.sin(proj), torch.cos(proj), x], dim=-1)


class Lff(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int,
                 mlp_layers: int,
                 sigma: float = 0.1,
                 ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        
        # Fourier features embedding 
        # Nota: La salida de ff_embedding tendrá dimensión: (hidden_dim // 2 * 2) + input_dim = hidden_dim + input_dim.
        self.layers.append(ff_embedding(in_features=input_dim,
                                        out_features=hidden_dim,
                                        sigma=sigma))
        
        # Ajustamos la dimensión de entrada de la primera capa lineal del MLP debido al concat del embedding
        mlp_in_dim = hidden_dim + input_dim
        
        # MLP layers
        for i in range(mlp_layers - 1):            
            in_dim = mlp_in_dim if i == 0 else hidden_dim
            layer = nn.Linear(in_features=in_dim,
                              out_features=hidden_dim)
            init.zeros_(layer.bias)
            self.layers.append(nn.Sequential(layer, nn.ReLU()))
        
        # Last layer (salida a output_dim)
        layer = nn.Linear(in_features=hidden_dim,
                          out_features=output_dim)
        init.zeros_(layer.bias)
        self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Pass through all layers (Fixed self.layers() -> self.layers)
        for layer in self.layers:
            x = layer(x)
                
        return x