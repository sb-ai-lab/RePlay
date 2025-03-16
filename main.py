"""Main module"""

import os
import logging
import warnings
import yaml

from replay_benchmarks.utils.conf import load_config, seed_everything
from replay_benchmarks import TrainRunner, InferRunner

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


def main() -> None:
    config_dir = "./replay_benchmarks/configs"
    base_config_path = os.path.join(config_dir, "config.yaml")
    config = load_config(base_config_path, config_dir)
    logging.info("Configuration:\n%s", yaml.dump(config))

    seed_everything(config["env"]["SEED"])
    logging.info(f"Fixing seed: {config['env']['SEED']}")

    if config["mode"]["name"] == "train":
        runner = TrainRunner(config)
    elif config["mode"]["name"] == "infer":
        runner = InferRunner(config)
    else:
        raise ValueError(f"Unsupported mode: {config['mode']}")

    runner.run()


if __name__ == "__main__":
    main()
