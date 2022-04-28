import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from postreview.wnn.ntn_encodings import Thermometer, CircularEncoder

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class NTNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, tuple_size=32, encoding={}, seed=None):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.res = encoding.get("resolution", 512)
        self.input_dim = int(np.product(obs_space.shape)) * self.res
        self.policy_output_dim = torch.Size([num_outputs])
        self.vf_output_dim = torch.Size([1])
        self.tuple_size = tuple_size
        self.n_nodes = torch.ceil(
          torch.tensor(self.input_dim/self.tuple_size)
        ).type(torch.int).item()
        self.mapping = torch.randperm(
            self.input_dim,
            generator=torch.Generator().manual_seed(seed)
        )
        self.key_weights = torch.special.exp2(
            torch.arange(tuple_size).reshape(1, 1, -1)
        ).to(torch.long)

        self.mem_offsets = (2 ** tuple_size) * torch.arange(self.n_nodes).reshape((1, -1))

        self.policy_mem = nn.utils.skip_init(
            nn.EmbeddingBag,
            num_embeddings=self.n_nodes * 2 ** tuple_size,
            embedding_dim=self.policy_output_dim[0],
            mode='sum'
        )
        nn.init.zeros_(self.policy_mem.weight)

        # Greater std
        # import math
        # nn.init.constant_(
        #     self.policy_mem.weight.chunk(2, dim=1)[1],
        #     math.log(.5)
        # )

        self.vf_mem = nn.utils.skip_init(
            nn.EmbeddingBag,
            num_embeddings=self.n_nodes * 2 ** tuple_size,
            embedding_dim=self.vf_output_dim[0],
            mode='sum'
        )
        nn.init.zeros_(self.vf_mem.weight)

        enc_cls = Thermometer

        if encoding.get("enc_type", "thermometer") == "circular":
            enc_cls = CircularEncoder

        self.encoder = enc_cls(
            minimum=encoding.get("min", -1.0),
            maximum=encoding.get("max", 1.0),
            resolution=self.res
        )

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    def get_extra_state(self):
        return self.mapping

    def set_extra_state(self, state):
        self.mapping = state
        
    def _keys(self, x):
        return F.conv1d(
            x[:, self.mapping].unsqueeze(1),
            self.key_weights,
            stride=self.tuple_size
        ).squeeze(1)

    def _forward(self, x, mem):
        keys = self._keys(x)

        # RAM neurons are stacked along the first dimension of mem. Therefore,
        # it's necessary to offset the keys occording to which neuron we are
        # trying to access
        keys += self.mem_offsets

        return mem(keys)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        obs = obs.reshape(obs.shape[0], -1)
        self._last_flat_in = self.encoder.encode(obs).flatten(start_dim=1)

        self._features = self._forward(
            self._last_flat_in,
            self.policy_mem
        )

        return self._features, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"

        value = self._forward(
            self._last_flat_in,
            self.vf_mem
        )

        return torch.atleast_1d(value.squeeze())
