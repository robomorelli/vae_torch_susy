from pathlib import Path
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from utils import *

def main(save_model_path):

    sched = ASHAScheduler(metric="loss", mode="min")

    config = {
        "Nf_lognorm" : tune.choice([3]),
        "hidden_size" : tune.choice([20]),
        "latent_dim" : tune.choice([2, 3, 4]),
        "weight_KL_loss" : tune.choice([0.6]),
        "epochs" : tune.choice([100]),
        "lr" : tune.choice([0.003]),
        "act_fun" : tune.choice(['relu']),
        "model_name" : tune.choice(["vae_susy.h5"]),
        "batch_size": tune.choice([200, 400, 600, 100])
    }

    analysis = tune.run(tune.with_parameters(train),
                        scheduler=sched,
                        #stop={
                        #      "training_iteration": 100},
                        resources_per_trial={"cpu": 10, "gpu":1},
                        num_samples=128,
                        checkpoint_at_end=False,
                        #checkpoint_dir = save_model_path

                        config=config)

    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))

if __name__ == "__main__":
  # ip_head and redis_passwords are set by ray cluster shell scripts
  os.environ["ip_head"] = '10.141.1.13:6379'
  os.environ["redis_password"] = '5241590000000000'
  print(os.environ["ip_head"], os.environ["redis_password"])
  ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])

  save_model_path = Path(model_results_path)

  if not (save_model_path.exists()):
      print('creating path')
      os.makedirs(save_model_path)

  main(save_model_path)
