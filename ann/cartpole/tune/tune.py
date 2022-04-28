import os

import numpy as np
import random

from ntn_neurocomputing.wnn.ntn_model import NTNModel

import ray
from ray import tune
from ray.rllib.models import ModelCatalog


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
            "env": "CartPole-v1",
            "framework": "torch",
            "num_workers": 16,
            "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
            "lr": tune.grid_search([0.0001, 0.0003, 0.001]),
            "observation_filter": "MeanStdFilter",
            "gamma": tune.grid_search([0.97, 0.98, 1.0]),
            "train_batch_size": tune.grid_search([2_000, 4_000, 8_000]),
            "num_sgd_iter": tune.grid_search([1, 3, 5]),
            "sgd_minibatch_size": tune.grid_search([128, 256]),
            "model": {
                "fcnet_hiddens": [32],
                "fcnet_activation": "linear"
            },
        },
        stop={ 'timesteps_total': 1_000_000 },
        checkpoint_freq=5,
        checkpoint_at_end=True
    )

    ray.shutdown()
