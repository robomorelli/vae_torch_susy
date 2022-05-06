from __future__ import print_function

import os
import torch

import ray_class
from ray_class import tune
from ray_class.tune.schedulers import ASHAScheduler
from pathlib import Path

from config import *
from utils import *

class Vae(tune.Trainable):
  def __init__(self, train_dict, data_folder, save_model_path):
      self.train_dict = train_dict
      self.data_folder = data_folder
      self.save_model_path = save_model_path

  def setup(self, config):
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    self.device = torch.device("cuda" if use_cuda else "cpu")
    print('forcing to cpu')
    self.device = 'cpu'
    print(' DEVICE is {}'.format(self.device))
    #self.train_loader, self.test_loader = get_data_loaders()
    self.train_data, self.train_loader, self.test_loader = load_data_train_eval(self.data_folder, self.train_dict,
                                                                                feat_selected=['met', 'mt', 'mct2'])
    #self.model = ConvNet().to(self.device)
    input_size = train_data[0][0].size()[0]
    self.model = initialize_model(train_dict, input_size).to(self.device)

    #############################
    #############################
    #############################
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


if __name__ == "__main__":
  # ip_head and redis_passwords are set by ray cluster shell scripts
  print(os.environ["ip_head"], os.environ["redis_password"])
  ray_class.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])
  sched = ASHAScheduler(metric="mean_accuracy")

  train_dict = {"batch_size": 200,
                "hidden_size": 20,
                "latent_dim": 3,
                "weight_KL_loss": 0.6,
                "Nf_lognorm": 3,
                "epochs": 100,
                "lr": 0.003,
                "act_fun": 'relu',
                "model_name": 'vae_susy.h5'}

  save_model_path = Path(model_results_path)

  if not (save_model_path.exists()):
      print('creating path')
      os.makedirs(save_model_path)

  analysis = tune.run(Vae(train_dict, data_folder=train_val_test, save_model_path=save_model_path),
                      scheduler=sched,
                      stop={"mean_accuracy": 0.99,
                            "training_iteration": 100},
                      resources_per_trial={"cpu":10},
                      num_samples=128,
                      checkpoint_at_end=False,
                      config={"lr": tune.uniform(0.001, 1.0),
                              "momentum": tune.uniform(0.1, 0.9),
                             "use_gpu": True})
  print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
