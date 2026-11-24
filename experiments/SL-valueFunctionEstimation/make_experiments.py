################################################################################################################
# This script performs the supervised training of the V:S^Ds -> R value function.
# We make a trick getting the real V function as grounth truth from the enviroment.
# The idea is to see how good the NN archs are without the sampling component from RL
################################################################################################################

# Imports base
import os
import time
import json
import inspect
import itertools
import warnings
import tomllib
import torch
import gymnasium as gym
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# Baselines
from src.models.mlp import Mlp
from src.models.simba import Simba, Simba_ablation
from src.models.simbav2 import Simbav2
from src.models.siren import Siren, SirenV2, SirenSimba, SirenSimbaV2
from src.models.f_decomposition import F_decomposition
from src.models.learned_ff import Lff, Clff

# Utils funcs
from src.utils import model_hyperparameters, show_dict, make_exp_supervised, get_data

# Manual device selection: "cpu", "cuda", "cuda:0", etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)

# V ground truth path
ROOT = Path("./V_truth")

CURRENT_DIR = os.getcwd()
TOML_PATH = os.path.join(CURRENT_DIR, "configs", "experiment.toml")
with open(TOML_PATH, "rb") as f:
    config = tomllib.load(f)

BATCH_SIZE = config.get("BATCH_SIZE", 64)

# Results directory
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

models_dict = {
    # Baseline
    #"MLP": Mlp,

    # INRs
    #"SIREN": Siren,

    # Learned fourier features
    "Learned-FF": Lff,
    #"C-Learned-FF": Clff,
}

# Environments definition
envs_dict = {
    "MountainCar": "MountainCar-v0",
    "Pendulum": "Pendulum-v1"
}

for model_name, model_class in models_dict.items():
    print(f"||||||| Training (Supervised) with {model_name} on {device} |||||||", flush=True)

    for env_key, env_id in envs_dict.items():
        print(f"\n--- Environment: {env_key} ({env_id}) ---", flush=True)

        env = gym.make(env_id)
        state, info = env.reset()
        input_dim = 2
        output_dim = 1

        # Model hyperparameters setup...
        model_config = model_hyperparameters(name=model_name,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            CURRENT_DIR=CURRENT_DIR)

        normalized_config = {k: v if isinstance(v, list) else [v] for k, v in model_config.items()}
        keys = list(normalized_config.keys())
        combinations = list(itertools.product(*normalized_config.values()))

        # Get the DataFrame containing states and V*
        df_data = get_data(env_key, env)

        # Separate inputs (X) and target (Y) as PyTorch tensors on the selected device
        X = torch.tensor(df_data.iloc[:, :-1].values, dtype=torch.float32, device=device)
        Y_ground_truth = torch.tensor(df_data.iloc[:, -1].values, dtype=torch.float32, device=device)

        exp_params = inspect.signature(make_exp_supervised).parameters
        device_kwarg = {"device": device} if "device" in exp_params else {}

        for i, values in enumerate(combinations):
            current_config = dict(zip(keys, values))
            print(f"{'-'*10} Model hyperparameters ({env_key}) {'-'*10}", flush=True)
            show_dict(current_config)
            print("-"*43, flush=True)

            lr = current_config.pop("lr")
            valid_keys = inspect.signature(model_class.__init__).parameters.keys()
            filtered_config = {k: v for k, v in current_config.items() if k in valid_keys}

            start_t = time.perf_counter()

            # Run single supervised training run
            loss, V_estimation = make_exp_supervised(
                model_class=model_class,
                X=X,
                Y=Y_ground_truth,
                lr=lr,
                batch_size=64,
                #batch_size=BATCH_SIZE,
                **device_kwarg,
                **filtered_config
            )

            elapsed_time = time.perf_counter() - start_t
            print(f"Elapsed time: {elapsed_time}", flush=True)

            df_loss = pd.DataFrame({
                "loss": loss
            }).T

            timestamp = datetime.now().strftime("%Y-%m-%d")

            # Save loss
            output_dir = os.path.join(RESULTS_DIR, model_name.upper(), env_key)
            os.makedirs(output_dir, exist_ok=True)
            save_name = f"{model_name}_{env_key}_config_{i}_{timestamp}"

            table = pa.Table.from_pandas(df_loss)
            full_config = {"lr": lr, **current_config, "environment": env_id, "device": str(device), "elapsed_time": elapsed_time}
            metadata = {b"hyperparameters": json.dumps(full_config).encode("utf-8")}
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
            pq.write_table(table, os.path.join(output_dir, f"{save_name}.parquet"))

            # Save V-estimation and its configuration TOML (overwriting if it already exists)
            estimation_dir = os.path.join(CURRENT_DIR, "V_estimation", model_name.upper(), env_key)
            os.makedirs(estimation_dir, exist_ok=True)

            base_filename = f"V_est_config_{i}"

            v_est_np = V_estimation.detach().cpu().numpy()
            np.save(os.path.join(estimation_dir, f"{base_filename}.npy"), v_est_np)

            full_config_to_save = {
                "lr": lr,
                **current_config,
                "environment": env_id,
                "device": str(device),
                "elapsed_time": elapsed_time
            }

            toml_path = os.path.join(estimation_dir, f"{base_filename}.toml")
            with open(toml_path, "w", encoding="utf-8") as f:
                for k, v in full_config_to_save.items():
                    f.write(f"{k} = {repr(v)}\n")

            # End
            print(f"Model {model_name} on {env_key} (Config {i}) successfully trained and V_estimation saved!\n", flush=True)

        env.close()