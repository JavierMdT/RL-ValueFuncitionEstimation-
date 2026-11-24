import math
import torch
import torch.nn as nn
import torch.nn.init as init

################################### Learned Fourier Features ###################################

class LffEmbedding(nn.Module):
    """
    Computes learned Fourier feature embedding: sin(W @ x + b)
    with W_ij ~ N(0, pi * beta / d) and b_j ~ U(-pi, pi).
    """

    def __init__(self, in_features: int, out_features: int, beta: float = 1.0):
        super().__init__()
        std = math.sqrt(math.pi * beta / in_features)
        self.W = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.b = nn.Parameter(
            torch.empty(out_features).uniform_(-math.pi, math.pi)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.nn.functional.linear(x, self.W, self.b))


class Lff(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        mlp_layers: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            LffEmbedding(
                in_features=input_dim,
                out_features=hidden_dim,
                beta=beta,
            )
        )

        for i in range(mlp_layers - 1):
            in_dim = hidden_dim
            layer = nn.Linear(in_features=in_dim, out_features=hidden_dim)
            init.zeros_(layer.bias)
            self.layers.append(nn.Sequential(layer, nn.ReLU()))

        layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


################################### Concatenaded Learned Fourier Features ###################################

class ff_embedding(nn.Module):
    """
    Computes Fourier feature embedding: [sin(B @ x) || cos(B @ x) || x]
    with B_ij ~ N(0, pi * beta / d).
    """

    def __init__(self, in_features: int, out_features: int, beta: float = 1.0):
        super().__init__()
        std = math.sqrt(math.pi * beta / in_features)
        initial_B = torch.randn(out_features // 2, in_features) * std
        self.B = nn.Parameter(initial_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(proj), torch.cos(proj), x], dim=-1)


class Clff(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        mlp_layers: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(
            ff_embedding(
                in_features=input_dim,
                out_features=hidden_dim,
                beta=beta,
            )
        )

        mlp_in_dim = (hidden_dim // 2) * 2 + input_dim

        for i in range(mlp_layers - 1):
            in_dim = mlp_in_dim if i == 0 else hidden_dim
            layer = nn.Linear(in_features=in_dim, out_features=hidden_dim)
            init.zeros_(layer.bias)
            self.layers.append(nn.Sequential(layer, nn.ReLU()))

        layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x