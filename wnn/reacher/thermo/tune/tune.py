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
        name="tune",
        local_dir=os.path.abspath(os.path.dirname(__file__)),
        resume="AUTO",
        config={
            "env": "ReacherBulletEnv-v0",
            "framework": "torch",
            "num_workers": 16,
            "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
            "lr": tune.grid_search([0.00003, 0.00007, 0.0001, 0.0002]),
            "observation_filter": "MeanStdFilter",
            "gamma": tune.grid_search([0.98, 0.99]),
            "train_batch_size": tune.grid_search([1_000, 2_000, 4_000]),
            "num_sgd_iter": tune.grid_search([1, 2, 3]),
            "sgd_minibatch_size": tune.grid_search([128, 256]),
            "model": {
                "custom_model": "ntn_model",
                "custom_model_config": {
                    "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
                    "tuple_size": tune.grid_search([5, 10, 15]),
                    "encoding": {
                        "enc_type": "thermometer",
                        "resolution": tune.grid_search([300, 400, 500]),
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
