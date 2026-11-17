import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def train_supervised(device, model, X, Y, optimizer, batch_size=None, epochs=50):
    """
    Performs supervised training. Uses mini-batches if batch_size is an integer,
    or full-batch training if batch_size is None.
    
    Returns:
        tuple: (epoch_losses, final_predictions)
    """
    model.to(device)
    model.train()
    
    X = X.to(device)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    Y = Y.to(device)
    
    criterion = nn.MSELoss()
    epoch_losses = []

    # Mini-batch training mode
    if batch_size is not None:
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            batch_losses = []
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                
            epoch_losses.append(np.mean(batch_losses))
            
    # Full-batch training mode
    else:
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, Y)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
    # Compute final predictions over the entire input space X
    model.eval()
    with torch.no_grad():
        final_predictions = model(X).cpu()
            
    return epoch_losses, final_predictions