from ml_collections import config_dict


def get_config(alg_type):
    config = {
        "ucb": config_dict.ConfigDict(
            {
                "model": "UCB",
                "params": config_dict.ConfigDict({"exploration_coef": 2.0, "sample": False}),
                "opt": config_dict.ConfigDict({"do_opt": True, "param_borders": {"coef": [0, 10]}}),
            }
        ),
        "wilson": config_dict.ConfigDict(
            {
                "model": "Wilson",
                "params": config_dict.ConfigDict({}),
                "opt": config_dict.ConfigDict({"do_opt": False, "param_borders": {}}),
            }
        ),
        "random": config_dict.ConfigDict(
            {
                "model": "RandomRec",
                "params": config_dict.ConfigDict({"seed": 42, "distribution": "uniform"}),
                "opt": config_dict.ConfigDict({"do_opt": False, "param_borders": {}}),
            }
        ),
        "lightfm": config_dict.ConfigDict(
            {
                "model": "LightFMWrap",
                "params": config_dict.ConfigDict({"random_state": 42, "loss": "warp", "no_components": 128}),
                "opt": config_dict.ConfigDict({"do_opt": True, "param_borders": {"no_components": [8, 512]}}),
            }
        ),
    }[alg_type]

    config.opt_params = config_dict.ConfigDict({"subset_borders": [0, 300000], "val_size": 0.3, "budget": 40})

    config.spark_params = config_dict.ConfigDict({"spark_memory": 4, "shuffle_partitions": 16})

    config.test_size = 0.3
    config.behavior_policy = "random"
    config.data_path = "/home/hdilab01/hdiRecSys/zozo_full/open_bandit_dataset"  # your path to the Open Bandit Dataset

    return config
