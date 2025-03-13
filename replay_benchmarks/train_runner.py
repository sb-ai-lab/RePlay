import logging
import os
import json
import pickle
from pathlib import Path

import optuna
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from replay_benchmarks.base_runner import BaseRunner
from replay.metrics import (
    OfflineMetrics,
    Recall,
    Precision,
    MAP,
    NDCG,
    HitRate,
    MRR,
    Coverage,
    Surprisal,
)
from replay.metrics.torch_metrics_builder import metrics_to_df
from replay.models.nn.sequential import SasRec, Bert4Rec
from replay.models.nn.optimizer_utils import FatOptimizerFactory
from replay.models.nn.sequential.callbacks import (
    ValidationMetricsCallback,
    PandasPredictionCallback,
)
from replay.models.nn.sequential.postprocessors import RemoveSeenItems
from replay.models.nn.sequential.sasrec import (
    SasRecTrainingDataset,
    SasRecValidationDataset,
    SasRecPredictionDataset,
)
from replay.models.nn.sequential.bert4rec import (
    Bert4RecTrainingDataset,
    Bert4RecValidationDataset,
    Bert4RecPredictionDataset,
)


class TrainRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.item_count = None
        self.raw_test_gt = None
        self.seq_val_dataset = None
        self.seq_test_dataset = None

        # Loggers
        self.log_dir = (Path(config["paths"]["log_dir"]) / self.dataset_name / self.model_save_name)
        self.csv_logger = CSVLogger(save_dir=self.log_dir / "csv_logs")
        self.tb_logger = TensorBoardLogger(save_dir=self.log_dir / "tb_logs")

        self._check_paths()

    def _check_paths(self):
        """Ensure all required directories exist."""
        required_paths = [
            self.config["paths"]["log_dir"],
            self.config["paths"]["checkpoint_dir"],
            self.config["paths"]["results_dir"],
        ]
        for path in required_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _initialize_model(self, trial=None):
        """Initialize the model based on configuration or Optuna trial parameters."""
        model_config = {
            "tensor_schema": self.tensor_schema,
        }

        if trial:
            search_space = self.config["optuna"]["search_space"][self.model_name]

            model_config.update({
                "block_count": trial.suggest_categorical("block_count", search_space["block_count"]),
                "head_count": trial.suggest_categorical("head_count", search_space["head_count"]),
                "hidden_size": trial.suggest_categorical("hidden_size", search_space["hidden_size"]),
                "max_seq_len": trial.suggest_categorical("max_seq_len", search_space["max_seq_len"]),
                "dropout_rate": trial.suggest_float("dropout_rate", float(min(search_space["dropout_rate"])), float(max(search_space["dropout_rate"])), step=0.05),
                "loss_type": trial.suggest_categorical("loss_type", search_space["loss_type"]),
            })

            optimizer_factory = FatOptimizerFactory(
                learning_rate=trial.suggest_float("learning_rate", float(min(search_space["learning_rate"])), float(max(search_space["learning_rate"])), log=True),
                weight_decay=trial.suggest_float("weight_decay", float(min(search_space["weight_decay"])), float(max(search_space["weight_decay"])), log=True),
            )
        else:
            optimizer_factory = FatOptimizerFactory(
                learning_rate=self.model_cfg["training_params"]["learning_rate"],
                weight_decay=self.model_cfg["training_params"].get("weight_decay", 0.0),
            )

        model_config.update(self.model_cfg["model_params"])

        if "sasrec" in self.model_name.lower():
            return SasRec(**model_config, optimizer_factory=optimizer_factory)
        elif "bert4rec" in self.model_name.lower():
            if self.config.get("acceleration"):
                if self.config["acceleration"].get("model"):
                    model_config.update(self.config["acceleration"]["model"])
            return Bert4Rec(**model_config, optimizer_factory=optimizer_factory)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def _prepare_dataloaders(
        self,
        seq_train_dataset,
        seq_validation_dataset,
        seq_validation_gt,
        seq_test_dataset,
    ):
        """Initialize dataloaders for training, validation, and testing."""
        logging.info("Preparing dataloaders...")

        dataset_mapping = {
            "sasrec": (
                SasRecTrainingDataset,
                SasRecValidationDataset,
                SasRecPredictionDataset,
            ),
            "bert4rec": (
                Bert4RecTrainingDataset,
                Bert4RecValidationDataset,
                Bert4RecPredictionDataset,
            ),
        }

        pad_idx = self.tensor_schema.item_id_features.item().cardinality - 1

        datasets = dataset_mapping.get(self.model_name.lower())
        if not datasets:
            raise ValueError(
                f"Unsupported model type for dataloaders: {self.model_name}"
            )

        TrainingDataset, ValidationDataset, PredictionDataset = datasets
        common_params = {
            "batch_size": self.model_cfg["training_params"]["batch_size"],
            "num_workers": self.model_cfg["training_params"]["num_workers"],
            "pin_memory": True,
        }

        train_dataloader = DataLoader(
            dataset=TrainingDataset(
                seq_train_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
                padding_value=pad_idx,
            ),
            shuffle=True,
            **common_params,
        )
        val_dataloader = DataLoader(
            dataset=ValidationDataset(
                seq_validation_dataset,
                seq_validation_gt,
                seq_train_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
                padding_value=pad_idx,
            ),
            **common_params,
        )
        val_pred_dataloader = DataLoader(
            dataset=PredictionDataset(
                seq_validation_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
                padding_value=pad_idx,
            ),
            **common_params,
        )
        prediction_dataloader = DataLoader(
            dataset=PredictionDataset(
                seq_test_dataset,
                max_sequence_length=self.model_cfg["model_params"]["max_seq_len"],
            ),
            **common_params,
        )

        return (
            train_dataloader,
            val_dataloader,
            val_pred_dataloader,
            prediction_dataloader,
        )

    def _load_dataloaders(self):
        """Loads data and prepares dataloaders."""
        logging.info("Preparing datasets for training.")
        train_events, validation_events, validation_gt, test_events, test_gt = (
            self.load_data()
        )
        self.validation_gt = validation_gt
        self.test_events = test_events
        self.raw_test_gt = test_gt

        (
            train_dataset,
            train_val_dataset,
            val_dataset,
            val_gt_dataset,
            test_dataset,
            test_gt_dataset,
        ) = self.prepare_datasets(
            train_events, validation_events, validation_gt, test_events, test_gt
        )
        self.item_count = train_dataset.item_count

        (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        ) = self.prepare_seq_datasets(
            train_dataset,
            train_val_dataset,
            val_dataset,
            val_gt_dataset,
            test_dataset,
            test_gt_dataset,
        )
        self.seq_val_dataset = seq_validation_dataset
        self.seq_test_dataset = seq_test_dataset

        return self._prepare_dataloaders(
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        )

    def calculate_metrics(self, predictions, ground_truth, test_events=None):
        """Calculate and return the desired metrics based on the predictions."""
        top_k = self.config["metrics"]["ks"]
        base_metrics = [
            Recall(top_k),
            Precision(top_k),
            MAP(top_k),
            NDCG(top_k),
            MRR(top_k),
            HitRate(top_k),
        ]

        diversity_metrics = []
        if test_events is not None:
            diversity_metrics = [
                Coverage(top_k),
                Surprisal(top_k),
            ]

        all_metrics = base_metrics + diversity_metrics
        metrics_results = OfflineMetrics(
            all_metrics, 
            query_column=self.user_column, 
            item_column=self.item_column, 
            rating_column="score",
        )(
            predictions,
            ground_truth,
            test_events,
        )

        return metrics_to_df(metrics_results)

    def save_model(self, trainer, best_model):
        """Save the best model checkpoint to the specified directory."""
        save_path = os.path.join(
            self.config["paths"]["checkpoint_dir"],
            f"{self.model_save_name}_{self.dataset_name}",
        )
        torch.save(
            {
                "model_state_dict": best_model.state_dict(),
                "optimizer_state_dict": trainer.optimizers[0].state_dict(),
                "config": self.model_cfg,
            },
            f"{save_path}/{self.model_save_name}_checkpoint.pth",
        )

        self.tokenizer.save(f"{save_path}/sequence_tokenizer")
        logging.info(f"Best model saved at: {save_path}")

    def _run_optuna_optimization(self, train_dataloader, val_dataloader):
        """Runs Optuna hyperparameter optimization"""
        optuna_dir = (
            Path(self.config["paths"]["checkpoint_dir"])
            / "optimization"
            / f"{self.model_save_name}_{self.dataset_name}"
        )
        optuna_dir.mkdir(parents=True, exist_ok=True)

        def objective(trial):
            """Objective function for Optuna"""
            model = self._initialize_model(trial)

            checkpoint_callback = ModelCheckpoint(
                dirpath=optuna_dir / f"optuna_trial_{trial.number}",
                save_top_k=1,
                monitor="ndcg@10",
                mode="max",
            )
            early_stopping = EarlyStopping(monitor="ndcg@10", patience=4, mode="max")
            validation_metrics_callback = ValidationMetricsCallback(
                metrics=["ndcg", "recall", "map"],
                ks=[10, 20],
                item_count=self.item_count,
                postprocessors=[RemoveSeenItems(self.seq_val_dataset)],
            )

            trainer = L.Trainer(
                max_epochs=20,
                callbacks=[
                    checkpoint_callback,
                    early_stopping,
                    validation_metrics_callback,
                ],
                logger=[self.csv_logger, self.tb_logger],
                devices=1,
            )

            trainer.fit(model, train_dataloader, val_dataloader)
            val_metrics = trainer.callback_metrics
            return val_metrics.get("ndcg@10", 0)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config["optuna"]["n_trials"],
            timeout=self.config["optuna"]["timeout"],
        )

        best_params_path = optuna_dir / "best_params.json"
        study_pickle_path = optuna_dir / "study.pkl"
        study_history_path = optuna_dir / "study_history.json"

        with open(best_params_path, "w") as f:
            json.dump(study.best_params, f, indent=4)

        with open(study_pickle_path, "wb") as f:
            pickle.dump(study, f)

        study_history = [
            {"trial": t.number, "params": t.params, "value": t.value}
            for t in study.trials
        ]
        with open(study_history_path, "w") as f:
            json.dump(study_history, f, indent=4)

        logging.info(f"Best hyperparameters: {study.best_params}")

    def run(self):
        """Execute the training pipeline."""
        train_dataloader, val_dataloader, val_pred_dataloader, prediction_dataloader = (
            self._load_dataloaders()
        )
        if self.config["mode"]["name"] == "optimize":
            logging.info("Running Optuna hyperparameter optimization...")
            self._run_optuna_optimization(train_dataloader, val_dataloader)
        else:
            logging.info("Initializing model...")
            model = self._initialize_model()

            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(
                    self.config["paths"]["checkpoint_dir"],
                    f"{self.model_save_name}_{self.dataset_name}",
                ),
                save_top_k=1,
                verbose=True,
                monitor="ndcg@10",
                mode="max",
            )

            early_stopping = EarlyStopping(
                monitor="ndcg@10",
                patience=self.model_cfg["training_params"]["patience"],
                mode="max",
                verbose=True,
            )

            validation_metrics_callback = ValidationMetricsCallback(
                metrics=self.config["metrics"]["types"],
                ks=self.config["metrics"]["ks"],
                item_count=self.item_count,
                postprocessors=[RemoveSeenItems(self.seq_val_dataset)],
            )

            profiler = SimpleProfiler(
                dirpath=self.csv_logger.log_dir, filename="simple_profiler"
            )

            devices = [int(self.config["env"]["CUDA_VISIBLE_DEVICES"])]
            trainer = L.Trainer(
                max_epochs=self.model_cfg["training_params"]["max_epochs"],
                callbacks=[
                    checkpoint_callback,
                    early_stopping,
                    validation_metrics_callback,
                ],
                logger=[self.csv_logger, self.tb_logger],
                profiler=profiler,
                precision=self.model_cfg["training_params"]["precision"],
                devices=devices,
            )

            logging.info("Starting training...")
            if self.config["mode"]["profiler"]["enabled"]:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True,
                ) as prof:
                    trainer.fit(model, train_dataloader, val_dataloader)
                logging.info(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=self.config["mode"]["profiler"].get("row_limit", 10),
                    )
                )
                prof.export_chrome_trace(
                    os.path.join(
                        self.config["paths"]["log_dir"],
                        f"{self.model_save_name}_{self.dataset_name}_profile.json",
                    )
                )
            else:
                trainer.fit(model, train_dataloader, val_dataloader)

            if self.model_name.lower() == "sasrec":
                best_model = SasRec.load_from_checkpoint(
                    checkpoint_callback.best_model_path
                )
            elif self.model_name.lower() == "bert4rec":
                best_model = Bert4Rec.load_from_checkpoint(
                    checkpoint_callback.best_model_path
                )
            self.save_model(trainer, best_model)

            logging.info("Evaluating on val set...")
            pandas_prediction_callback = PandasPredictionCallback(
                top_k=max(self.config["metrics"]["ks"]),
                query_column=self.user_column,
                item_column=self.item_column,
                rating_column="score",
                postprocessors=[RemoveSeenItems(self.seq_val_dataset)],
            )
            L.Trainer(
                callbacks=[pandas_prediction_callback],
                inference_mode=True,
                devices=devices,
            ).predict(
                best_model, dataloaders=val_pred_dataloader, return_predictions=False
            )

            result = pandas_prediction_callback.get_result()
            recommendations = (
                self.tokenizer.query_and_item_id_encoder.inverse_transform(result)
            )
            val_metrics = self.calculate_metrics(recommendations, self.validation_gt)
            logging.info(val_metrics)
            recommendations.to_parquet(
                os.path.join(
                    self.config["paths"]["results_dir"],
                    f"{self.model_save_name}_{self.dataset_name}_val_preds.parquet",
                ),
            )
            val_metrics.to_csv(
                os.path.join(
                    self.config["paths"]["results_dir"],
                    f"{self.model_save_name}_{self.dataset_name}_val_metrics.csv",
                ),
            )

            logging.info("Evaluating on test set...")
            pandas_prediction_callback = PandasPredictionCallback(
                top_k=max(self.config["metrics"]["ks"]),
                query_column=self.user_column,
                item_column=self.item_column,
                rating_column="score",
                postprocessors=[RemoveSeenItems(self.seq_test_dataset)],
            )
            L.Trainer(
                callbacks=[pandas_prediction_callback],
                inference_mode=True,
                devices=devices,
            ).predict(best_model, dataloaders=prediction_dataloader, return_predictions=False)

            result = pandas_prediction_callback.get_result()
            recommendations = (self.tokenizer.query_and_item_id_encoder.inverse_transform(result))
            test_metrics = self.calculate_metrics(recommendations, self.raw_test_gt, self.test_events)
            logging.info(test_metrics)
            recommendations.to_parquet(
                os.path.join(
                    self.config["paths"]["results_dir"],
                    f"{self.model_save_name}_{self.dataset_name}_test_preds.parquet",
                ),
            )
            test_metrics.to_csv(
                os.path.join(
                    self.config["paths"]["results_dir"],
                    f"{self.model_save_name}_{self.dataset_name}_test_metrics.csv",
                ),
            )
