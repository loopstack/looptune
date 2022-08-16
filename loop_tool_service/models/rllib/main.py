# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import (train, test, get_data_loaders,
                                             ConvNet)

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument("--use-gpu", action="store_true", default=True, help="enables CUDA training")
parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")

# Below comments are for documentation purposes only.
# yapf: disable
# __trainable_example_begin__
class TrainMNIST(tune.Trainable):
  def setup(self, config):
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    self.device = torch.device("cuda" if use_cuda else "cpu")
    self.train_loader, self.test_loader = get_data_loaders()
    self.model = ConvNet().to(self.device)
    self.optimizer = optim.SGD(
        self.model.parameters(),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9))

  def step(self):
    self.current_ip()
    train(self.model, self.optimizer, self.train_loader, device=self.device)
    acc = test(self.model, self.test_loader, self.device)
    return {"mean_accuracy": acc}

  def save_checkpoint(self, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    torch.save(self.model.state_dict(), checkpoint_path)
    return checkpoint_path

  def load_checkpoint(self, checkpoint_path):
    self.model.load_state_dict(torch.load(checkpoint_path))

  # this is currently needed to handle Cori GPU multiple interfaces
  def current_ip(self):
    import socket
    hostname = socket.getfqdn(socket.gethostname())
    self._local_ip = socket.gethostbyname(hostname)
    return self._local_ip 

from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT
from datetime import datetime


if __name__ == "__main__":
  # ip_head and redis_passwords are set by ray cluster shell scripts
  ray_address = os.environ["RAY_ADDRESS"] if "RAY_ADDRESS" in os.environ else "auto"
  head_node_ip = os.environ["HEAD_NODE_IP"] if "HEAD_NODE_IP" in os.environ else "127.0.0.1"
  redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "5241590000000000"
  print(ray_address, head_node_ip, redis_password)

  ray.init(address=ray_address, _node_ip_address=head_node_ip, _redis_password=redis_password)
  # ray.init()

  sched = ASHAScheduler(metric="mean_accuracy", mode="max")
  analysis = tune.run(TrainMNIST,
                      scheduler=sched,
                      stop={"mean_accuracy": 0.99,
                            "training_iteration": 10},
                      resources_per_trial={"cpu":10, "gpu": 1},
                      # num_samples=4,
                      # metric="episode_reward_mean", # "final_performance",
                      # mode="max",
                      checkpoint_at_end=True,
                      config={
                        'num_workers': 60,
                        "num_gpus": 2,
                        # "lr": tune.uniform(0.001, 1.0),
                        # "momentum": tune.uniform(0.1, 0.9),
                        "use_gpu": True
                      },
                      callbacks=[ WandbLoggerCallback(
                            project="loop_tool_agent",
                            api_key_file=str(LOOP_TOOL_ROOT) + "/wandb_key.txt",
                            log_config=False)]
)
  print("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode="max"))
