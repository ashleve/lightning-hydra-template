# import os, sys
# sys.path.insert(1, os.path.join(sys.path[0], ".."))
# print(os.path.abspath(os.curdir))

import hydra
import wandb


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config):
    wandb.init(project="env_tests")
    wandb.finish()


if __name__ == "__main__":
    main()
