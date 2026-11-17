import torch
from torch import nn
from torch.nn import functional as F

class Mlp(nn.Module):
    '''
    Standar neural network class
    '''
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 hidden_dim:int,
                 n_hidden_layers:int) -> None:
        '''
        Args:
            input_dim: |N| -> Dimensions of the state set 
            output_dim: |A| -> Dimensions of the action set 
            hidden_dim: MLP hidden/latent dimension
            n_hidden_layers: Nº of hidden layers (Total layers = n_hidden_layers + 2)
            
        '''
        super().__init__()
        
        # Hyperparameters assertions
        assert(n_hidden_layers >= 2), "[ERROR] n_hidden_layers must be at least 2"
        assert(input_dim >= 1), "[ERROR] input_dim must be higher than 0"
        assert(output_dim >= 1), "[ERROR] output_dim must be higher than 0"
        assert(hidden_dim >= 1), "[ERROR] hidden_dim must be higher than 0"

        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(in_features = input_dim,
                                     out_features = hidden_dim))
    
        # Mid layers
        for i in range(n_hidden_layers-2):
            self.layers.append(nn.Linear(in_features=hidden_dim,
                                         out_features=hidden_dim))
        
        # Last layer
        self.layers.append(nn.Linear(in_features = hidden_dim,
                                    out_features = output_dim)) 



    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Method that performs the forward pass
        
        Args:
            x: Data tensor, in this case, a batch of states.
        '''
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)