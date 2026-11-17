from collections import deque, namedtuple
import random
import torch
from torch import nn
import math 
from typing import Protocol, Any
from itertools import count 
import tomllib
import os
import numpy as np

# Named tuple to store the transitions 
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyperparameters
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TOML_PATH = os.path.join(CURRENT_DIR, "hyperparameters", "dqn.toml")
with open(TOML_PATH, "rb") as f:
    config = tomllib.load(f)
GAMMA = config["GAMMA"]            # Discount factor 
EPS_START = config["EPS_START"]    # Epsilon start value
EPS_END = config["EPS_END"]        # Epsilon end value 
EPS_DECAY = config["EPS_DECAY"]    # Epsilon decay ration 
TAU = config["TAU"]                # Target network update rate 

class SupportedEnv(Protocol):
    '''
    Static duck typing to generlize the env typing, just to not raise pylance errors
    '''
    action_space: Any
    
    def reset(self) -> tuple[Any, dict]:
        ...
    def step(self, *args) -> tuple:
        ...

class ReplayMemory(object):
    '''
    Cyclic buffer to store/sample transitions collected/to try by the agent 
    '''
    def __init__(self,
                 capacity:int) -> None:
        '''
        Args:
            capacity: max capacitiy for the deque to store transitions
        '''
        self.memory = deque([], maxlen=capacity)  

    def push(self,
             state: torch.Tensor,
             action: torch.Tensor,
             next_state: torch.Tensor | None,
             reward: torch.Tensor ) -> None:
        '''
        Method to push new items into the memory
        
        Args:
            state: state from where the action was taken
            action: action performed in the transition
            next_state: state reached after the action
            reward: reward associated with the state-action pair (s,a)
        '''
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self,
               batch_size:int) -> list:
        '''
        Method to sample a batch from memory for trainingd
        '''
        
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
t:int = 0

# Standar inference
def select_action(state: torch.Tensor,
                  policy_net:nn.Module ,
                  environment: SupportedEnv,
                  device: str) -> torch.Tensor:
    global t
    '''
    Selects the action to perform base on the ϵ-greedy algorithm 
    
    Args:
        state: state tensor
        policy_net: policy network for action selection 
        environment: traninig environment 
        device: target device for compuation ('cuda' or 'cpu')
        
    Returns:device
        - `torch.Tensor()` with the 'BATCH_SIZE' selected actions 
    '''
    
    # Threshold value 
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t/EPS_DECAY)
    t += 1
    
    # Threshold evaluation 
    if sample > eps_threshold: # Exploitation
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
        
    else: # Exploration
        return torch.tensor([[environment.action_space.sample()]],
                              device=device,
                              dtype=torch.long)

def optimize_model(memory: ReplayMemory,
                   device: str,
                   policy_net: nn.Module,
                   target_net: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   batch_size: int) -> torch.Tensor:
    '''
    Performs one optimize step of the algorithm
    
    Args:
        memory: the memory of the agent 
        device: target device for compuation ('cuda' or 'cpu')
        policy_net: policy network for action selection 
        target_net: target network for evaluation 
        optimizer: torch optimizer to update network weights
        batch_size: replay memory batch size
    '''
    
    # The agent dont start optimizing until the memory has at least BATCH_SIZE trasitions
    if len(memory) < batch_size:
        return torch.tensor([0])
    
    # Sample and transpose 
    transitions = memory.sample(batch_size = batch_size) # [t1=[s,a,n_s,r], t2=[s,a,n_s,r], ...]
    batch = Transition(*zip(*transitions))               # transition = [s=[...], a=[...], n_s[...], r=[...]]
    
    # Mask and list of non terminal transitions 
    non_f_mask = torch.tensor([s is not None for s in batch.next_state],
                              device = device,
                              dtype = torch.bool)
    non_f_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # Make tensor of the rest of the inputs
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s) and index the selected actions
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for the non terminal transitions 
    next_state_values = torch.zeros(batch_size, device = device)
    with torch.no_grad():
        next_state_values[non_f_mask] = target_net(non_f_next_states).max(1).values
        
    # Compute expected value Q(s,a) = r + gamma * max(Q(s'))
    expected_state_action_value = (next_state_values * GAMMA) + reward_batch 
    
    # Compute huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_value.unsqueeze(dim = 1))
    
    # Optimizer step
    optimizer.zero_grad()
    loss.backward()
    ##torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss


# Train function 
def train(num_episodes: int,
          device: str,
          policy_net: nn.Module,
          target_net: nn.Module,
          optimizer: torch.optim.Optimizer,
          environment: SupportedEnv,
          memory: ReplayMemory,
          batch_size: int,
          show_results:bool) -> tuple:
    '''
    Trains the agent by the DQN algorithm
    
    Args:
        num_episodes: Number of episodes to train the agent
        device: target device for compuation ('cuda' or 'cpu')
        policy_net: policy network for action selection 
        target_net: target network for evaluation 
        optimizer: torch optimizer to update network weights
        enviroment: traninig environment 
        memory: the memory of the agent 
        batch_size: sample memory batch size 
        show_results: flag to show results every 5% iterations done
    '''
    
    if show_results: print(f"[INFO] Training started...")
    episodes_durations = []
    episodes_loss = []
    
    assert(num_episodes>0), "num_episodes must be higher than 0"
    assert(device in ["cuda", "cpu"]), "device must be 'cpu' or 'cuda'"
    
    step = num_episodes/20
    
    # Loop through episodes
    for i_episode in range(num_episodes):
        # Reset and initial state
        state, info = environment.reset()
        state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        
        # Loop through time 
        ep_loss = []
        for t in count():
            # Perform action and save results
            action = select_action(state = state, policy_net = policy_net, 
                                   environment = environment, device = device)
            observation, reward, terminated, truncated, _ = environment.step(action.item())
            reward = torch.tensor([reward], device = device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)
            
            # Save the new transition observed
            memory.push(state = state, action = action, next_state = next_state, reward = reward)
            
            # Move the agent to next state and optimize 
            state = next_state
            loss = optimize_model(memory = memory, device = device, policy_net = policy_net,
                                  target_net = target_net, optimizer = optimizer, batch_size = batch_size)
            ep_loss.append(loss.cpu().item())
            # Target soft update
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            for key in policy_net_state_dict.keys():
                target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU)*target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)
            
            # Stop condition and save episode results
            if done:
                episodes_durations.append(t+1)
                episodes_loss.append(np.array([ep_loss]).mean())
                break
        if show_results and (i_episode % step == 0):
            print(f"Episode {i_episode} results: {t+1}")
            
    if show_results: print(f"[INFO] Training ended!")
    return (episodes_durations, episodes_loss)

            
            
            
            