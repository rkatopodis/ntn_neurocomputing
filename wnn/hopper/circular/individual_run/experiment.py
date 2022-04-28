import os

import numpy as np
import random

from ntn_neurocomputing.wnn.ntn_model import NTNModel

import ray
from ray import tune
from ray.rllib.models import ModelCatalog

import pybullet_envs

if __name__ == "__main__":
    ray.init()

    ModelCatalog.register_custom_model("ntn_model", NTNModel)

    seed = 12345678

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    analysis = tune.run(
        "PPO",
        name="experiment",
        local_dir=os.path.abspath(os.path.dirname(__file__)),
        resume="AUTO",
        num_samples=20,
        config={
            "env": "HopperBulletEnv-v0",
            "framework": "torch",
            "num_workers": 17,
            "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
            "lr": 0.0004,
            "observation_filter": "MeanStdFilter",
            "gamma": 0.98,
            "vf_clip_param": 3_000,
            "train_batch_size": 1_000,
            "num_sgd_iter": 1,
            "sgd_minibatch_size": 512,
            "batch_mode": "complete_episodes",
            "model": {
                "custom_model": "ntn_model",
                "custom_model_config": {
                    "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
                    "tuple_size": 8,
                    "encoding": {
                        "enc_type": "circular",
                        "resolution": 32,
                        "min": -1.5,
                        "max": 1.5
                    }
                },
            },
        },
        stop={ 'timesteps_total': 1_000_000 },
        checkpoint_freq=5,
        checkpoint_at_end=True
    )

    ray.shutdown()