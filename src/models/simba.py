from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

class Rsnorm(nn.Module):
    '''
    It performs the Running Statistics Normalization 
    '''
    
    def __init__(self,
                 input_dim:int,
                 epsilon: float = 1e-8) -> None:
        super().__init__()
        '''
        Args:
            input_dim: |N| -> Dimensions of the state set 
            epsilon: Stability number for square root at normalization 
        '''
        
        self.epsilon = epsilon

        # RSNorm statistics 
        self.register_buffer('running_mean', torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer('running_var', torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer('t', torch.zeros(1, dtype=torch.float32))
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ### Running Statistics ###
        if self.training:
            with torch.no_grad():
                self.t += 1
                
                # Mean difference
                delta_t =  x.mean(0) - self.running_mean
                
                # Statistics update
                self.running_mean = self.running_mean + 1/self.t * delta_t
                self.running_var = ((self.t-1)/self.t) * (self.running_var + 1/self.t * delta_t**2) 
        
        ### Normalization ###
        x = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
        return x

class Simba(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 hidden_dim:int,
                 n_hidden_blocks:int) -> None:
        super().__init__()
        '''
        Args:
            input_dim: |N| -> Dimensions of the state set 
            output_dim: |A| -> Dimensions of the action set 
            hidden_dim: Linear layers hidden/latent dimension       
            n_hidden_blocks: Number of hidden layers    
        '''
    
        # Embedding
        self.embedding = nn.Linear(in_features=input_dim,
                                   out_features=hidden_dim)
        
        # Rsnorm
        self.rsnorm = Rsnorm(input_dim=input_dim)
        
        # Hidden layers 
        self.hidden = nn.ModuleList()
        for i in range(n_hidden_blocks):
            self.hidden.append(nn.Sequential(nn.LayerNorm(normalized_shape=hidden_dim),
                                             nn.Linear(in_features=hidden_dim,
                                                       out_features=4 * hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(in_features=4 * hidden_dim,
                                                       out_features=hidden_dim)))

        
        self.last_norm = nn.LayerNorm(normalized_shape=hidden_dim)

        # Classifier 
        self.classifier = (nn.Linear(in_features=hidden_dim,
                                     out_features=output_dim))     
        
    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        '''
        Method that performs the forward pass
        
        Args:
            obs: Observation tensor (a batch of states with dim |S|).
        '''
        
        obs_norm = self.rsnorm(obs)
        
        ### Embedding ###
        z = self.embedding(obs_norm)
        
        ### Hidden layers ###
        for l in self.hidden:
            z = z + l(z) # Residual conexion 
        
        # Final output
        return self.classifier(self.last_norm(z))
    
class Simba_ablation(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 hidden_dim:int,
                 n_hidden_blocks:int,
                 _InvBottleNeck:str,
                 _Rsnorm:bool = True,
                 _Lnorm:bool = True,
                 _ResidualConexion:bool = True,
                 ) -> None:
        super().__init__()
        '''
        Args:
            input_dim: |N| -> Dimensions of the state set 
            output_dim: |A| -> Dimensions of the action set 
            hidden_dim: Linear layers hidden/latent dimension       
            n_hidden_blocks: Number of hidden layers    
            _InvBottleNeck: Ablation study flag for the Inverse Bottleneck 
            _Rsnorm: Ablation study flag for Running Statistics
            _Lnorm: Ablation study flag for Layer Norm
            _ResidualConexion: Ablation study flag for Residual Conexion
        '''
    
        self._ResidualConexion = _ResidualConexion
        self._Rsnorm = _Rsnorm
        self._Lnorm = _Lnorm
        
        # Embedding
        self.embedding = nn.Linear(in_features=input_dim,
                                   out_features=hidden_dim)
        # Rsnorm
        if _Rsnorm:
            self.rsnorm = Rsnorm(input_dim=input_dim)
        
        # Latent space expansion feature 
        _Expansion = 1 if _InvBottleNeck in (1,2) else 4
        
        # Hidden layers 
        self.hidden = nn.ModuleList()
        for i in range(n_hidden_blocks):
            if _Lnorm:
                if _Rsnorm:
                    sequential_tmp = nn.Sequential(nn.LayerNorm(normalized_shape=hidden_dim),
                                                   nn.Linear(in_features=hidden_dim,
                                                            out_features=_Expansion * hidden_dim),
                                                   nn.ReLU())
                else:
                    sequential_tmp = nn.Sequential(nn.Linear(in_features=hidden_dim,
                                                             out_features=_Expansion * hidden_dim),
                                                   nn.ReLU())
                if _InvBottleNeck in (0, 1):
                    sequential_tmp.append(nn.Linear(in_features=_Expansion * hidden_dim,
                                                    out_features=hidden_dim))
                self.hidden.append(sequential_tmp)
            else:    
                self.hidden.append(nn.Sequential(nn.Linear(in_features=hidden_dim,
                                                           out_features=_Expansion * hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=_Expansion * hidden_dim,
                                                           out_features=hidden_dim)))

        
        if _Lnorm:
            self.last_norm = nn.LayerNorm(normalized_shape=hidden_dim)

        # Classifier 
        self.classifier = (nn.Linear(in_features=hidden_dim,
                                     out_features=output_dim))     
        
    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        '''
        Method that performs the forward pass
        
        Args:
            obs: Observation tensor (a batch of states with dim |S|).
        '''
        
        if self._Rsnorm:
            obs = self.rsnorm(obs)
        
        ### Embedding ###
        z = self.embedding(obs)
        
        ### Hidden layers ###
        if self._ResidualConexion:
            for l in self.hidden:
                z = z + l(z) # Residual conexion 
        else:
            for l in self.hidden:
                z = l(z) # Residual conexion 
        
        # Final output
        if self._Lnorm:
            return self.classifier(self.last_norm(z))
        else: return self.classifier(z)
            