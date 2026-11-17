import torch
from torch import nn
from torch.nn import functional as F
from typing import cast


################################################ LAYERS ################################################
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

class HiddenBlocks(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        '''
        Performs the hidden blocks code (MLP + LERP + L2Norm)
        
        Args:
            hidden_dim: Linear layers hidden dimension
        '''
        
        ## MLP 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            Scale(dim=4 * hidden_dim),
            nn.ReLU(),
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

class ComputeLogits(nn.Module):
    # ! NOT IN USE 
    def __init__(self,
                 A_dim:int,
                 n_atom:int,
                 d_h:int):
        super().__init__()
        '''
        Performs the learnable linear interpolation
        
        Args:
            dim: Dimension of alpha vector 
        '''
        
        self.W = nn.Parameter(torch.randn(A_dim, n_atom, d_h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bf, anf -> ban', x, self.W)
    

################################################ MODELS ################################################
class Simbav2(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 hidden_dim:int,
                 n_hidden_blocks:int,
                 C:float) -> None:
        super().__init__()
        '''
        Args:
            input_dim: |N| -> Dimensions of the state set 
            output_dim: |A| -> Dimensions of the action set 
            hidden_dim: Linear layers hidden/latent dimension       
            n_hidden_blocks: Number of hidden layers 
            C: Shift value 
            
        Returns: 
            q_values: torch.tensor with shape (BATCH x |A|) 
        '''
        
        self.input_dim = input_dim
        
        # RSNorm statistics 
        self.RSNorm = Rsnorm(input_dim=input_dim)
        
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
            self.hidden_blocks.append(
                HiddenBlocks(hidden_dim=hidden_dim))
        
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
            obs: Observation tensor (a batch of states with dim |S|).
        '''
        
        # Making sure we have batch dimension
        obs.reshape(-1, self.input_dim)
        
        ### RSNorm ###
        obs_norm = self.RSNorm(obs)
        
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