from typing import cast
import os 
import numpy as np
import math
import inspect

import torch
from torch import nn

from src.models import mlp, simba, siren
from src.utils import model_hyperparameters

class VirtualActionMatrix:
    '''
    Class that simulates the Action space 
    '''
    
    def __init__(self, cardinalities: list[int], device: torch.device, dtype: torch.dtype):
        self.cardinalities = cardinalities
        self.device = device
        self.dtype = dtype
        
        self.num_a = math.prod(cardinalities)
        self.shape = (self.num_a, len(cardinalities))

    # A[:,j] or A[j] sVirtualActionMatrixintaxis
    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, tuple) and isinstance(key[0], slice) and isinstance(key[1], int):
            j = key[1]
        elif isinstance(key, int):
            j = key
        else:
            raise NotImplementedError("Solo se soporta indexación por columnas: A[:, j] o A[j]")
            
        if j < 0 or j >= len(self.cardinalities):
            raise IndexError("Índice de dimensión fuera de rango")
            
        # Genereting a_j 
        c_j = self.cardinalities[j]
        inner_repeats = math.prod(self.cardinalities[j+1:]) if j + 1 < len(self.cardinalities) else 1
        outer_repeats = math.prod(self.cardinalities[:j]) if j > 0 else 1
        
        base = torch.arange(c_j, device=self.device, dtype=self.dtype)
        return base.repeat_interleave(inner_repeats).repeat(outer_repeats)

class F_decomposition(nn.Module):
    def __init__(self,
                 model_name: str,
                 rank: int,
                 input_dim: int,
                 dim_a:int,
                 cardinalities: list,
                 env_limits: dict
                 ) -> None:
        super().__init__()
        '''
        
        Args:
            model_name: Backbone name  
            rank: the rank of the low rank aproximation
            input_dim: dim(S)
            dim_a: dim(A)
            cardinalities: Action subspaces cardinalities -> [|A1|, |A2|, ...]
            env_limits: Enviroments limits
        '''
        
        # Models dict 
        models = {
            "mlp":mlp.Mlp,
            "simba":simba.Simba,
            "siren":siren.Siren,
            "sirenmlp":siren.SirenMLP,
        }
        
        # Backbone 
        backbone = models[model_name.lower()]
        self.rank = rank
        self.states_sub_nets = nn.ModuleList()
        self.actions_sub_nets = nn.ModuleList()
        self.cardinalities = cardinalities
        
        
        # Select the hyperparameters for the backbone model
        CURRENT_DIR = os.getcwd()
        backbone_config = model_hyperparameters(name=model_name.upper(),
                                                input_dim=2,
                                                output_dim=1,
                                                CURRENT_DIR=CURRENT_DIR)
        backbone_config.pop("lr", None)
        backbone_config["env_limits"] = env_limits
        
        valid_backbone_keys = inspect.signature(backbone.__init__).parameters.keys()
        
        # Defining a INR per dimension of the Q tensor (d = dim(A) + dim(S))
        for i in range(input_dim):
            
            # Select the limit of the current state dim
            backbone_config["env_limits"] = {
                "min":env_limits["min"][i],
                "max":env_limits["max"][i]
            }
            filtered_backbone_config = {k: v for k, v in backbone_config.items() if k in valid_backbone_keys}
            self.states_sub_nets.append(backbone(**filtered_backbone_config))
            
        for i in range(dim_a):
            # Select the limit of the current action dim
            backbone_config["env_limits"] = {
                "min":0,
                "max":cardinalities[i]-1
            }
            filtered_backbone_config = {k: v for k, v in backbone_config.items() if k in valid_backbone_keys}
            self.actions_sub_nets.append(backbone(**filtered_backbone_config))    
            
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        
            '''
            Computes Q(s) with tensor decomposition using INRs of the factors
            '''
                    
            # States & actions INRs
            state_subnets = self.states_sub_nets
            action_subnets = self.actions_sub_nets
            
            # Action space tensor & K
            A = VirtualActionMatrix(self.cardinalities, device=s.device, dtype=s.dtype)
            k_vals = torch.arange(self.rank, device=s.device, dtype=s.dtype)
            
            # Shapes
            batch_size = s.shape[0]
            num_k = k_vals.shape[0]
            num_a = math.prod(self.cardinalities)

            # Compute the states products
            states_product = torch.ones(batch_size, num_k, device=s.device, dtype=s.dtype)
            for i, net in enumerate(state_subnets):
                s_i = s[:, i].unsqueeze(1).expand(-1, num_k)
                k_S = k_vals.unsqueeze(0).expand(batch_size, -1)
                
                s_i_flat = s_i.reshape(-1, 1)
                k_S_flat = k_S.reshape(-1, 1)
                
                # Concat
                net_input = torch.cat([s_i_flat, k_S_flat], dim=1)  
                
                states_output = net(net_input)
                states_product *= states_output.reshape(batch_size, num_k)

            # Compute the action products
            actions_product = torch.ones(num_k, num_a, device=s.device, dtype=s.dtype)
            for j, net in enumerate(action_subnets):
                # Extraemos la dimensión j para TODAS las acciones -> forma: [num_a]
                a_j = A[:,j] 
                
                # Expandimos para cruzar con K -> forma: [num_k, num_a]
                a_j_exp = a_j.unsqueeze(0).expand(num_k, -1)
                k_A_exp = k_vals.unsqueeze(1).expand(-1, num_a)
                
                # Aplanamos a [num_k * num_a, 1]
                a_j_flat = a_j_exp.reshape(-1, 1)
                k_A_flat = k_A_exp.reshape(-1, 1)
                
                # Concatenamos -> forma: [num_k * num_a, 2]
                net_input = torch.cat([a_j_flat, k_A_flat], dim=1)
                
                # Evaluamos la subred j (encargada de la dimensión j de la acción)
                actions_output = net(net_input)
                
                # Restauramos la forma [num_k, num_a] y multiplicamos
                actions_product *= actions_output.reshape(num_k, num_a)

            # Compute the products tensor Batch x K X |A|
            products = states_product.unsqueeze(2) * actions_product.unsqueeze(0)        
        
            return torch.sum(products, dim=1) # Final output Batch x |A|


class F_decomposition_V2(nn.Module):
    def __init__(self,
                 model_name: str,
                 rank: int,
                 input_dim: int,
                 dim_a: int,
                 cardinalities: list,
                 env_limits: dict
                 ) -> None:
        super().__init__()
        '''
        Args:
            model_name: Backbone name  
            rank: the rank of the low rank approximation
            input_dim: dim(S)
            dim_a: dim(A)
            cardinalities: Action subspaces cardinalities -> [|A1|, |A2|, ...]
            env_limits: Environments limits
        '''
        
        # Models dict 
        models = {
            "mlp": mlp.Mlp,
            "simba": simba.Simba,
            "siren": siren.Siren,
            "sirenmlp": siren.SirenMLP,
        }
        
        # Backbone 
        backbone = models[model_name.lower()]
        self.rank = rank
        self.cardinalities = cardinalities
        
        # Structures to save the d * K subnets
        self.states_sub_nets = nn.ModuleList()
        self.actions_sub_nets = nn.ModuleList()
        
        # Select the hyperparameters for the backbone model 
        CURRENT_DIR = os.getcwd()
        backbone_config = model_hyperparameters(name=model_name.upper(),
                                                input_dim=1,
                                                output_dim=1,
                                                CURRENT_DIR=CURRENT_DIR)
        backbone_config.pop("lr", None)
        backbone_config["env_limits"] = env_limits
        
        valid_backbone_keys = inspect.signature(backbone.__init__).parameters.keys()
        
        # 1. Creating dim(s) * k subnets
        for i in range(input_dim):
            rank_nets = nn.ModuleList()
            for k in range(rank):
                # Limits
                dim_config = backbone_config.copy()
                dim_config["env_limits"] = {
                    "min": env_limits["min"][i],
                    "max": env_limits["max"][i]
                }
                filtered_config = {k: v for k, v in dim_config.items() if k in valid_backbone_keys}
                rank_nets.append(backbone(**filtered_config))
            self.states_sub_nets.append(rank_nets)
            
        # 2. Creatung dim(a) * k subnets 
        for j in range(dim_a):
            rank_nets = nn.ModuleList()
            for k in range(rank):
                # Limits
                dim_config = backbone_config.copy()
                dim_config["env_limits"] = {
                    "min": 0,
                    "max": cardinalities[j] - 1
                }
                filtered_config = {k: v for k, v in dim_config.items() if k in valid_backbone_keys}
                rank_nets.append(backbone(**filtered_config))
            self.actions_sub_nets.append(rank_nets)    
            
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        '''
        Computes Q(s) with tensor decomposition using univariate factor INRs
        '''
        batch_size = s.shape[0]
        num_a = math.prod(self.cardinalities)
        
        # Action space tensor & K
        A = VirtualActionMatrix(self.cardinalities, device=s.device, dtype=s.dtype)
        
        # Compute the states products [Batch, K]
        states_product = torch.ones(batch_size, self.rank, device=s.device, dtype=s.dtype)
        for i, rank_nets in enumerate(self.states_sub_nets):
            # Getting i-th dimnesion from every state of the batch -> [Batch, 1]
            s_i = s[:, i].unsqueeze(1)
            
            for k, net in enumerate(rank_nets):
                # Univariant evaluation
                s_output = net(s_i)  # [Batch, 1]
                states_product[:, k] *= s_output.squeeze(1)

        # Compute the action products [K, |A|]
        actions_product = torch.ones(self.rank, num_a, device=s.device, dtype=s.dtype)
        for j, rank_nets in enumerate(self.actions_sub_nets):
            # Getting j-th dimnesion from every action possible -> [Batch, 1]
            a_j = A[:, j].unsqueeze(1)
            
            for k, net in enumerate(rank_nets):
                # Univariant evaluation
                a_output = net(a_j)  # [|A|, 1]
                actions_product[k, :] *= a_output.squeeze(1)

        # Compute the products tensor Batch x K x |A|
        products = states_product.unsqueeze(2) * actions_product.unsqueeze(0)        
    
        return torch.sum(products, dim=1)  # Final output Batch x |A|