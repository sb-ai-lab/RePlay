from ml_collections import config_dict

def get_config(alg_type):
    config = {
        "lin_ucb": config_dict.ConfigDict({
                "model": "LinUCB",
                "params": config_dict.ConfigDict({
                        "eps": -2.0,
                        "alpha": 1.0,
                        "regr_type": "disjoint"
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": True,
                        "param_borders": {
                                "eps": [-10, 10],
                                "alpha": [0.001, 10]
                            }
                    })
            }),

        "log_ucb": config_dict.ConfigDict({
                "model": "LogUCB",
                "params": config_dict.ConfigDict({
                        "eps": -10.0,
                        "alpha": 1.0,
                        "random_state": 42
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": True,
                        "param_borders": {
                                "eps": [-10, 10],
                                "alpha": [0.01, 10]
                            }
                    })
            }),

        "lin_ts": config_dict.ConfigDict({
                "model": "LinTS",
                "params": config_dict.ConfigDict({
                        "nu": 0.0,
                        "alpha": 1.0,
                        "regr_type": "disjoint",
                        "random_state": 42
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": True,
                        "param_borders": {
                                "nu": [1, 10],
                                "alpha": [100, 1000]
                            }
                    })
            }),

        "log_ts": config_dict.ConfigDict({
                "model": "LogTS",
                "params": config_dict.ConfigDict({
                        "eps": 0.0,
                        "alpha": 1e-9,
                        "regr_type": "disjoint",
                        "random_state": 42
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": True,
                        "param_borders": {
                                "eps": [0.0, 10.0],
                                "alpha": [100, 1000]
                            }
                    })
            }),

        "ucb": config_dict.ConfigDict({
                "model": "UCB",
                "params": config_dict.ConfigDict({
                        "exploration_coef": 2.0,
                        "sample": False
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": True,
                        "param_borders": {
                                "coef": [0, 10]
                            }
                    })
            }),

        "kl_ucb": config_dict.ConfigDict({
                "model": "KL_UCB",
                "params": config_dict.ConfigDict({
                        "exploration_coef": 0.0
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": True,
                        "param_borders": {"coef": [0, 3]}
                    })
            }),

        "wilson": config_dict.ConfigDict({
                "model": "Wilson",
                "params": config_dict.ConfigDict({}),
                "opt": config_dict.ConfigDict({
                        "do_opt": False,
                        "param_borders": {}
                    })
            }),

        "random": config_dict.ConfigDict({
                "model": "RandomRec",
                "params": config_dict.ConfigDict({
                        "seed": 42,
                        "distribution": "uniform"
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": False,
                        "param_borders": {}
                    })
            }),

        "lightfm": config_dict.ConfigDict({
                "model": "LightFMWrap",
                "params": config_dict.ConfigDict({
                        "random_state": 42,
                        'loss': 'warp',
                        'no_components': 128
                    }),
                "opt": config_dict.ConfigDict({
                        "do_opt": True,
                        "param_borders": {"no_components": [8, 512]}
                    })
            })
    }[alg_type]

    config.opt_params = config_dict.ConfigDict({
            "subset_borders": [0, 300000],
            "val_size": 0.3,
            "budget": 40

        })

    config.behavior_policy = "random"
    config.data_path = None

    return config
