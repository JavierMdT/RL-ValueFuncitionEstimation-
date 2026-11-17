# Baselines 
from src.models.mlp import Mlp
from src.models.simba import Simba, Simba_ablation
from src.models.simbav2 import Simbav2
from src.models.siren import Siren, SirenV2, SirenMLP, SirenSimba, SirenSimbaV2
from src.models.f_decomposition import F_decomposition

# Standar libraries 
import gymnasium
import torch
import os 
import tomllib
import matplotlib.pyplot as plt
import numpy as np
import multiprocess as mp
import time 
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from datetime import datetime
import itertools
import inspect
import warnings
warnings.filterwarnings("ignore")

# Extra 
from src.utils import make_exps, make_exps_parallel, plot_results_centered, model_hyperparameters, show_dict, get_cardinalities

os.environ["CUDA_VISIBLE_DEVICES"] = "" # To avoid the warnings of the nvidia drivers of the cluster

# HYPERPARAMETERS
CURRENT_DIR = os.getcwd()
TOML_PATH = os.path.join(CURRENT_DIR, "configs", "experiment.toml")
with open(TOML_PATH, "rb") as f:
    config = tomllib.load(f)
NUM_EPISODES = config["NUM_EPISODES"]
MEMORY_CAPACITY = config["MEMORY_CAPACITY"]
BATCH_SIZE = config["BATCH_SIZE"]

# RL ENVIROMENT
env = gymnasium.make("CartPole-v1")
state, info = env.reset()
n_actions = int(env.action_space.n) #type:ignore
n_observations = len(state)


# EXPERIMENTS
models_dict = {
    #"MLP": Mlp,
    #"SIMBA": Simba,
    #"SIMBA_ABLATION": Simba_ablation,
    #"SIMBAV2":Simbav2,
    #"SIREN": Siren, 
    #"SIRENV2":SirenV2,
    #"SIRENMLP":SirenMLP,
    #"SIRENSIMBA":SirenSimba,    
    #"SIRENSIMBAV2":SirenSimbaV2, 
    "F_DECOMPOSITION":F_decomposition
}

import os

# Loop through models
for model_name, model_class in models_dict.items():
    print(f"||||||| Training with {model_name} |||||||")
    
    # Getting model configs 
    model_config = model_hyperparameters(name=model_name,
                                         input_dim=n_observations,
                                         output_dim=n_actions,
                                         CURRENT_DIR=CURRENT_DIR)
    
    # Defining the grid of the hyperparameters 
    normalized_config = {k: v if isinstance(v, list) else [v] for k, v in model_config.items()}
    keys = list(normalized_config.keys())
    combinations = list(itertools.product(*normalized_config.values()))
    
    # Loop through cofigurations of the models
    for i, values in enumerate(combinations):
        current_config = dict(zip(keys, values))
        
        # Printing hyperparameters
        print(f"{'-'*10} Model hyperparameters {'-'*10}")
        show_dict(current_config)
        print("-"*43)
        lr = current_config.pop("lr")
        current_config["env_limits"] = {
            "min": torch.tensor([-4.8, -0.5, -0.418, -0.9]),
            "max":torch.tensor([4.8, 0.5, 0.418, 0.9])
        }
        current_config["dim_a"] = 1
        current_config["cardinalities"] = get_cardinalities(env.action_space)
        
        valid_keys = inspect.signature(model_class.__init__).parameters.keys()
        filtered_config = {k: v for k, v in current_config.items() if k in valid_keys}
        
        # Training
        n_exps = 25
        start_t = time.perf_counter()
       
        durations, loss = make_exps_parallel(model=model_class, env=env, num_episodes=NUM_EPISODES,
                                             memory_capacity=MEMORY_CAPACITY, lr=lr, batch_size=BATCH_SIZE,
                                             num_runs=n_exps, **filtered_config)
        
        
        # Training elapsed time 
        elapsed_time = time.perf_counter() - start
        print(f"Elapsed time: {elapsed_time}")
        
        # Plotting results
        fig = plot_results_centered([durations, loss], title=f"{model_name} results (nº of experiments: {n_exps})")        
        
        # Timestamp & creating the dir to save results
        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_dir = os.path.join(CURRENT_DIR, f"tunning_exps/{model_name.upper()}", current_config["model_name"])

        os.makedirs(output_dir, exist_ok=True)
        
        # Saving the figure, model_name
        save_name = f"{model_name}_rank{current_config["rank"]}_{timestamp}"
        fig.savefig(os.path.join(output_dir, save_name+".png"))
        plt.close(fig)
        
        # Save results DataFrame and parquet metadata
        df_data = {}
        for idx in range(n_exps):
            df_data[f"duration_run_{idx}"] = durations[idx]
            df_data[f"loss_run_{idx}"] = loss[idx]
            
        df_results = pd.DataFrame(df_data)
        df_results.index.name = "episode"
        
        # Save the hyperparameters as metadata
        full_config = {"lr": lr, **current_config, "elapsed_time": elapsed_time}
        full_config.pop("env_limits")
        full_config.pop("cardinalities")
        table = pa.Table.from_pandas(df_results)
        metadata = {b"hyperparameters": json.dumps(full_config).encode("utf-8")}
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
        
        # Save the table
        pq.write_table(table, os.path.join(output_dir, save_name+".parquet"))
        
        print(f"Model {model_name} (Config {i}) successfully trained!\n")

print(f"All experiments all done!\n\n")