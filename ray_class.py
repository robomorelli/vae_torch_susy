from pathlib import Path
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from utils import *
import argparse

class Vae(tune.Trainable):

  def setup(self, config):
    #setup function is invoked once training starts
    #setup function is invoked once training starts
    #setup function is invoked once training starts
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    self.lr = config['lr']
    self.device = torch.device("cuda" if use_cuda else "cpu")
    self.config = config
    self.model_name = config["model_name"]
    self.feat_selected = config["feat_selected"]
    self.train_data, self.train_loader, self.test_loader = load_data_train_eval(self.feat_selected, config)
    self.input_size = self.train_data[0][0].size()[0]
    self.model = initialize_model(config, self.input_size).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

  def step(self):
    self.current_ip()
    val_loss = train_class(self.config, self.model, self.input_size, self.optimizer,
                           self.train_loader, self.test_loader, self.device, checkpoint_dir=None)
    result = {"loss":val_loss}
    #if detect_instance_preemption():
    #    result.update(should_checkpoint=True)
    #acc = test(self.model, self.test_loader, self.device)
    # don't call report here!
    return result

  def save_checkpoint(self, checkpoint_dir):
    print("this is the checkpoint dir {}".format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
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
  #os.environ["ip_head"] = '10.141.1.12:6379'
  #os.environ["redis_password"] = '5241590000000000'
  #ray.init(address='auto', _node_ip_address=args.address.split(":")[0], _redis_password=args.password)
  parser = argparse.ArgumentParser()
  parser.add_argument("--address", help="echo the string you use here")
  parser.add_argument("--password", help="echo the string you use here")
  args = parser.parse_args()
  print(args.address, args.password)
  ray.init(address='auto', _node_ip_address=args.address.split(":")[0], _redis_password=args.password)
  #ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])

  config = {
      "Nf_lognorm": tune.choice([3]),
      "hidden_size": tune.choice([20]),
      "latent_dim": tune.choice([2, 3, 4]),
      "weight_KL_loss": tune.choice([0.6]),
      "epochs": tune.choice([100]),
      "lr": tune.choice([0.003]),
      "act_fun": tune.choice(['relu']),
      "model_name": tune.choice(["vae_susy.h5"]),
      "batch_size": tune.choice([200, 400, 600, 100]),
      "feat_selected" : ['met', 'mt', 'mct2']
  }

  save_model_path = Path(model_results_path)

  if not (save_model_path.exists()):
      print('creating path')
      os.makedirs(save_model_path)

  sched = ASHAScheduler(metric="loss", mode="min")
  analysis = tune.run(Vae,
                      scheduler=sched,
                      stop={"training_iteration": 10 ** 16},
                      resources_per_trial={"cpu": 10, "gpu": 1},
                      num_samples=10,
                      checkpoint_at_end=True,
                      #checkpoint_freq=1,
                      local_dir="~/ray_results",
                      name="vae",
                      config=config)

  print("Best config is:", analysis.get_best_config(metric="loss"))
