
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()



class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        # ray.util.pdb.set_trace()
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])



from gym.spaces import Dict
from ray.rllib.utils.torch_utils import FLOAT_MIN



class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        # assert (
        #     isinstance(orig_space, Dict)
        #     and "action_mask" in orig_space.spaces
        #     and "observations" in orig_space.spaces
        # )
        self.action_mask = model_config['custom_model_config']['action_mask']
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        action_mask = torch.tensor(self.action_mask).to('cuda' if logits.is_cuda else 'cpu')
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state


    def value_function(self):
        return self.internal_model.value_function()



import numpy as np
from ray.rllib.models.torch.misc import (
    SlimFC,
    normc_initializer as torch_normc_initializer,
)
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
import ray

class TorchBatchNormModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization."""

    capture_index = 0

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in model_config['fcnet_hiddens']:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.ReLU,
                )
            )
            prev_layer_size = size
            # Add a batch norm layer.
            layers.append(nn.BatchNorm1d(prev_layer_size))

        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=self.num_outputs,
            initializer=torch_normc_initializer(0.01),
            activation_fn=None,
        )

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=torch_normc_initializer(1.0),
            activation_fn=None,
        )

        self._hidden_layers = nn.Sequential(*layers)
        self._hidden_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=is_training)
        self._hidden_out = self._hidden_layers(torch.flatten(input_dict["obs"], start_dim=1))
        logits = self._logits(self._hidden_out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])