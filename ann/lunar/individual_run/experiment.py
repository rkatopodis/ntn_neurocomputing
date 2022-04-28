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
        name="experiment",
        local_dir=os.path.abspath(os.path.dirname(__file__)),
        resume="AUTO",
        num_samples=20,
        config={
            "env": "LunarLander-v2",
            "framework": "torch",
            "num_workers": 2,
            "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
            "lr": 0.002,
            "observation_filter": "MeanStdFilter",
            "train_batch_size": 4_000,
            "num_sgd_iter": 1,
            "sgd_minibatch_size": 128,
            "model": {
                "fcnet_hiddens": [32],
            },
        },
        stop={ 'timesteps_total': 1_000_000 },
        checkpoint_freq=5,
        checkpoint_at_end=True
    )

    ray.shutdown()
