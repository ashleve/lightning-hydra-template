# THIS FILE WILL EXECUTE ALL RUNS SPECIFIED IN run_configs.yaml ONE AFTER THE OTHER
# YOU CAN ALSO SPECIFY PATH TO DIFFERENT FILE WITH RUN CONFIGS

from argparse import ArgumentParser
from train import train
import wandb
import yaml
import os


def main(project_config_path, run_configs_path):

    with open(project_config_path, "r") as ymlfile:
        project_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    with open(run_configs_path, "r") as ymlfile:
        run_configs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # EXECUTE ALL RUNS ONE AFTER THE OTHER
    for conf_name in run_configs:
        print()
        print("EXECUTING RUN:", conf_name)
        print("CONFIG:")
        for section in run_configs[conf_name]:
            print("  " + section + ":")
            for key in run_configs[conf_name][section]:
                print("    " + key + ":", run_configs[conf_name][section][key])
        print()

        train(project_config=project_config, run_config=run_configs[conf_name], use_wandb=True)
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--project_config_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "project_config.yaml"))
    parser.add_argument("-r", "--run_configs_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_configs.yaml"))
    args = parser.parse_args()

    main(project_config_path=args.project_config_path, run_configs_path=args.run_configs_path)
