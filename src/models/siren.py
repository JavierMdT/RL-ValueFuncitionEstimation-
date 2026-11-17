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
        
################################################ SIREN LAYERS ################################################
class SirenInputLayer(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 omega_0:int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        limit = 1.0 / in_features
        init.uniform_(self.linear.weight, a=-limit, b=limit) 
        init.uniform_(self.linear.bias, a=-limit, b=limit)
        self.omega_0 = omega_0
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
class SirenMidLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0): 
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        
        limit = math.sqrt(6.0 / in_features) / omega_0 
        init.uniform_(self.linear.weight, a=-limit, b=limit) 
        init.uniform_(self.linear.bias, a=-limit, b=limit) 
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
class SirenLastLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        limit = math.sqrt(6.0 / in_features) / omega_0
        init.uniform_(self.linear.weight, a=-limit, b=limit) 
        init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)
    
class SirenScaleLayer(nn.Module):
    '''
    Sine activation with different initialitation values 
    '''
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 omega_0:int,
                 layer_type:int):
        super().__init__()
        
        
        assert(layer_type in (1,2,3)), "Error: wrong layer type, must be in (1, 2, 3)"
        
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.s = nn.Parameter(torch.ones(out_features))
        
        # Initialiation for first layer
        if layer_type == 1:
            limit = 1.0 / in_features
            init.uniform_(self.linear.weight, a=-limit, b=limit) 
            init.uniform_(self.linear.bias, a=-limit, b=limit) 
            
        # Initialiation for mid layer
        elif layer_type == 2:
            limit = math.sqrt(6.0 / in_features) / omega_0 
            init.uniform_(self.linear.weight, a=-limit, b=limit) 
            init.uniform_(self.linear.bias, a=-limit, b=limit)  
            
        # Initialiation for last layer
        elif layer_type == 3:
            limit = math.sqrt(6.0 / in_features) / omega_0 
            init.uniform_(self.linear.weight, a=-limit, b=limit) 
            init.uniform_(self.linear.bias, a=-limit, b=limit) 
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w_x = F.linear(x, self.linear.weight, bias=None)
        return torch.sin((self.omega_0 * (w_x + self.linear.bias)) * self.s)
    
    
class MinMaxScale(nn.Module):
    def __init__(self,
                 min_points: list[float] | tuple[float, ...] | torch.Tensor,
                 max_points: list[float] | tuple[float, ...] | torch.Tensor,
                 eps: float = 1e-8) -> None:
        super().__init__()
        
        self.register_buffer("obs_min", torch.as_tensor(min_points, dtype=torch.float32))
        self.register_buffer("obs_max", torch.as_tensor(max_points, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * ((x - self.obs_min) / (self.obs_max - self.obs_min + self.eps)) - 1.0 #type:ignore
    
    
################################################ SIMBA LAYERS ################################################
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

class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        Performs the l2-normalization
        '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1, eps=1e-12)
          
class Shift(nn.Module):
    def __init__(self,
                 C:float):
        super().__init__()
        '''
        Shifts the input 
        
        Args:
            C: Scalar value of shifting 
        '''
        
        self.register_buffer('C_vector', torch.tensor([[float(C)]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_vec = cast(torch.Tensor, self.C_vector) # Cast type
        c_column = c_vec.expand(x.size(0), 1) # C vector 
        z = torch.cat((x, c_column), dim=1) # Shifted tensor 
        return z

class Scale(nn.Module):
    def __init__(self, 
                 dim: int):
        super().__init__()
        '''
        Performs the scaling 
        
        Args:
            dim: Dimension of s vector
        '''
        self.s = nn.Parameter(torch.ones(dim)) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.s

class HiddenBlocksSirenSimba(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 layer_type: int):
        super().__init__()
        '''
        Performs the hidden blocks code (MLP + LERP + L2Norm)
        
        Args:
            hidden_dim: Linear layers hidden dimension
            layer_type: Type of layer between first (1), mid (2) and last (3) to choose weights init
        '''
        
        ## MLP 
        self.mlp = nn.Sequential(
            SirenScaleLayer(in_features=hidden_dim,
                            out_features=4 * hidden_dim,
                            omega_0=30,
                            layer_type=layer_type),
            nn.Linear(4 * hidden_dim, hidden_dim),
            L2Norm()
        )
        
        # LERP & L2Norm
        self.alpha = nn.Parameter(torch.empty(hidden_dim))
        nn.init.uniform_(self.alpha, a=0.0, b=1.0)
        
        self.final_norm = L2Norm()

    def forward(self, h_original: torch.Tensor) -> torch.Tensor:
        h_transformed = self.mlp(h_original)
        
        h_lerp = (1 - self.alpha) * h_original + self.alpha * h_transformed
        
        return self.final_norm(h_lerp)

################################################ MODELS ################################################
class Siren(nn.Module):
    def __init__(self,
                 input_dim:int, 
                 output_dim:int,
                 hidden_dim:int,
                 n_hidden_layers:int,
                 omega_0:int = 30,
                 env_limits:dict = {
                     "min": torch.tensor([-4.8, -0.5, -0.418, -0.9]),
                     "max":torch.tensor([4.8, 0.5, 0.418, 0.9])}) -> None:
        super().__init__()
        
        '''
        Args:
            input_dim: dim(S) -> Dimensions of the state set 
            output_dim: |A| -> Number of elements of the action set 
            hidden_dim: Linear layers hidden/latent dimension       
            n_hidden_layers: Number of SIREN hidden layers 
            omega_0: Factor for the last layer init
            env_limits: Enviroment state dimensions limits 
        '''
        
        assert(n_hidden_layers>=3), "n_hidden_layers must be higher or equal to 3"
        
        # [-1,1] Scale layer
        self.Scale = MinMaxScale(min_points=env_limits["min"],
                                 max_points=env_limits["max"])
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(SirenInputLayer(in_features=input_dim,
                                           out_features=hidden_dim,
                                           omega_0=omega_0))
    
        # Mid layers
        for i in range(n_hidden_layers-2):
            self.layers.append(SirenMidLayer(in_features=hidden_dim,
                                             out_features=hidden_dim,
                                             omega_0=omega_0))
        
        # Last layer
        self.layers.append(SirenLastLayer(in_features=hidden_dim,
                                          out_features=output_dim,
                                          omega_0=omega_0))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Method that performs the forward pass
        
        Args:
            x: Data tensor, in this case, a batch of states.
        '''
        
        x = self.Scale(x)
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x
    
class SirenV2(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int,
                 n_hidden_layers: int,
                 _FirstLNorm: bool,
                 _Lnorm: bool,
                 _ResidualConexion:bool,
                 omega_0: int = 30,
                 env_limits:dict = {
                    "min": torch.tensor([-4.8, -0.5, -0.418, -0.9]),
                    "max":torch.tensor([4.8, 0.5, 0.418, 0.9])}) -> None:
        super().__init__()
        
        '''
        Args:
            input_dim: dim(S) -> Dimensions of the state set 
            output_dim: |A| -> Number of elements of the action set 
            hidden_dim: Linear layers hidden/latent dimension       
            n_hidden_layers: Number of SIREN hidden layers 
            _Lnorm: Flag to use or not the Lnorm layers
            _ResidualConexion: Flag to use or not the residual conexion 
            omega_0: Factor for the last layer init
            env_limits: Enviroment state dimensions limits 
        '''
        
        self.scale = MinMaxScale(min_points=env_limits["min"],
                                 max_points=env_limits["max"])
        self._ResidualConexion = _ResidualConexion
        self._Lnorm = _Lnorm
        
        # Input layer 
        if _FirstLNorm:
            self.input_norm = nn.LayerNorm(normalized_shape=input_dim)
        else:
            self.input_norm = nn.Identity()
            
        self.input_layer = SirenInputLayer(in_features=input_dim,
                                           out_features=hidden_dim,
                                           omega_0=omega_0)
    
        # Mid layers 
        self.mid_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_hidden_layers - 2)])
        self.mid_layers = nn.ModuleList([
            SirenMidLayer(in_features=hidden_dim, out_features=hidden_dim, omega_0=omega_0)
            for _ in range(n_hidden_layers - 2)
        ])
        
        # Last layer 
        self.last_layer = nn.Sequential(nn.LayerNorm(hidden_dim),
                                        SirenLastLayer(in_features=hidden_dim,
                                                        out_features=output_dim,
                                                        omega_0=omega_0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(self.input_norm(x))
        
        if self._Lnorm & self._ResidualConexion:
            for norm, layer in zip(self.mid_norms, self.mid_layers):
                x = x + layer(norm(x))
        elif self._Lnorm:
            for norm, layer in zip(self.mid_norms, self.mid_layers):
                x = layer(norm(x))
        elif self._ResidualConexion:
            for norm, layer in zip(self.mid_norms, self.mid_layers):
                x = x + layer(x)
            
        return self.last_layer(x)
       
class SirenSimba(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 hidden_dim:int,
                 n_hidden_blocks:int,
                 omega_0:int = 30) -> None:
        super().__init__()
        '''
        Args:
            input_dim: dim(S) -> Dimensions of the state set 
            output_dim: |A| -> Number of elements of the action set 
            hidden_dim: Linear layers hidden/latent dimension       
            n_hidden_blocks: Number of hidden layers    
            omega_0: Factor for the last layer init
        '''
        
        assert(n_hidden_blocks > 2), "At least 3 blocks "
    
        # Embedding
        self.embedding = nn.Linear(in_features=input_dim,
                                   out_features=hidden_dim)
        
        # [-1,1] Scale layer 
        self.Scale = MinMaxScale()
        
        # First block
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Sequential(nn.LayerNorm(normalized_shape=hidden_dim),

                                         SirenInputLayer(in_features=hidden_dim,
                                                         out_features=4 * hidden_dim,
                                                         omega_0=omega_0),
                                             
                                         nn.Linear(in_features=4 * hidden_dim,

                                                   out_features=hidden_dim)))
        # Hidden layers 
        for i in range(n_hidden_blocks-2):
            self.hidden.append(nn.Sequential(nn.LayerNorm(normalized_shape=hidden_dim),
                                             
                                             SirenMidLayer(in_features=hidden_dim,
                                                           out_features=4 * hidden_dim,
                                                           omega_0=omega_0),
                                             
                                             nn.Linear(in_features=4 * hidden_dim,
                                                       out_features=hidden_dim)))
        # Last block 
        self.hidden.append(nn.Sequential(nn.LayerNorm(normalized_shape=hidden_dim),                                         
                                         SirenLastLayer(in_features=hidden_dim,
                                                        out_features=4 * hidden_dim,
                                                        omega_0=omega_0),
                                         
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
            obs: Observation tensor (a batch of states).
        '''
        
        obs_norm = self.Scale(obs)
        
        ### Embedding ###
        z = self.embedding(obs_norm)
        
        ### Hidden layers ###
        for l in self.hidden:
            z = z + l(z) # Residual conexion 
        
        # Final output
        return self.classifier(self.last_norm(z))
        
class SirenSimbaV2(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 hidden_dim:int,
                 n_hidden_blocks:int,
                 C:float,
                 omega_0:int = 30) -> None:
        super().__init__()
        '''
        Args:
            input_dim: dim(S) -> Dimensions of the state set 
            output_dim: |A| -> Number of elements of the action set 
            hidden_dim: Linear layers hidden/latent dimension       
            n_hidden_blocks: Number of hidden layers 
            C: Shift value 
            omega_0: Factor for the last layer init
        '''
        
        self.input_dim = input_dim
        
        # [-1,1] Scale layer        
        self.Scale = MinMaxScale()
        
        # Shift & l2norm
        self.sl2norm = nn.Sequential(
            Shift(C=C),
            L2Norm()
        )
        
        # Embedding 
        self.embedding = nn.Sequential(nn.Linear(in_features=input_dim + 1,
                                                out_features=hidden_dim),
                                          Scale(dim=hidden_dim),
                                          L2Norm())
        
        # Hidden blocks
        self.hidden_blocks = nn.ModuleList()
        for i in range(n_hidden_blocks):
            if i == 0: layer_type = 1
            elif i == n_hidden_blocks - 1: layer_type = 3
            else: layer_type = 2
            self.hidden_blocks.append(
                HiddenBlocksSirenSimba(hidden_dim=hidden_dim,
                                       layer_type=layer_type))
        
        # Inference layer 
        self.inference_layer = nn.Sequential(
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
               Scale(dim=hidden_dim),
            nn.Linear(in_features=hidden_dim,
                      out_features=output_dim)
        )

        
    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        '''
        Method that performs the forward pass
        
        Args:
            obs: Observation tensor (a batch of states).
        '''
        
        # Making sure we have batch dimension
        obs.reshape(-1, self.input_dim)
        
        ### RSNorm ###
        obs_norm = self.Scale(obs)
        
        ### Shift & L2-Norm ###
        h = self.sl2norm(obs_norm)
        
        ### Embedding ###
        h = self.embedding(h) 
        
        ### Hidden blocks ### 
        for block in self.hidden_blocks:
            h = block(h)
        
        ### Inference layer ###
        q_values = (self.inference_layer(h))
        
        return q_values
    
