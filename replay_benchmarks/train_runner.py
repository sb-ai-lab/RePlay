import logging
import os
import yaml
from pathlib import Path

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
        self.log_dir = Path(config["paths"]["log_dir"]) / self.dataset_name / self.model_save_name
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

    def _initialize_model(self):
        """Initialize the model based on the configuration."""
        model_config = {
            "tensor_schema": self.tensor_schema,
            "optimizer_factory": FatOptimizerFactory(
                learning_rate=self.model_cfg["training_params"]["learning_rate"]
            ),
        }
        model_config.update(self.model_cfg["model_params"])

        if "sasrec" in self.model_name.lower():
            return SasRec(**model_config)
        elif "bert4rec" in self.model_name.lower():
            if self.config.get("acceleration"):
                if self.config["acceleration"].get("model"):
                    model_config.update(self.config["acceleration"]["model"])
            return Bert4Rec(**model_config)
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

        return train_dataloader, val_dataloader, prediction_dataloader

    def _load_dataloaders(self):
        """Loads data and prepares dataloaders."""
        logging.info("Preparing datasets for training.")
        train_events, validation_events, validation_gt, test_events, test_gt = (
            self.load_data()
        )
        self.raw_test_gt = test_gt

        train_dataset, val_dataset, val_gt_dataset, test_dataset, test_gt_dataset = (
            self.prepare_datasets(
                train_events, validation_events, validation_gt, test_events, test_gt
            )
        )
        self.item_count = train_dataset.item_count

        (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        ) = self.prepare_seq_datasets(
            train_dataset, val_dataset, val_gt_dataset, test_dataset, test_gt_dataset
        )
        self.seq_val_dataset = seq_validation_dataset
        self.seq_test_dataset = seq_test_dataset

        return self._prepare_dataloaders(
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        )

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate and return the desired metrics based on the predictions."""
        top_k = self.config["metrics"]["ks"]
        metrics_list = [
            Recall(top_k),
            Precision(top_k),
            MAP(top_k),
            NDCG(top_k),
            MRR(top_k),
            HitRate(top_k),
        ]
        metrics = OfflineMetrics(
            metrics_list, query_column="user_id", rating_column="score"
        )(predictions, ground_truth)
        return metrics_to_df(metrics)

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

    def run(self):
        """Execute the training pipeline."""
        train_dataloader, val_dataloader, prediction_dataloader = (
            self._load_dataloaders()
        )

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

        profiler = SimpleProfiler(dirpath = self.csv_logger.log_dir, filename = 'simple_profiler')

        trainer = L.Trainer(
            max_epochs=self.model_cfg["training_params"]["max_epochs"],
            callbacks=[checkpoint_callback, early_stopping, validation_metrics_callback],
            logger=[self.csv_logger, self.tb_logger],
            profiler=profiler,
            precision=self.model_cfg["training_params"]["precision"]
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
            best_model = SasRec.load_from_checkpoint(checkpoint_callback.best_model_path)
        elif self.model_name.lower() == "bert4rec":
            best_model = Bert4Rec.load_from_checkpoint(checkpoint_callback.best_model_path)
        self.save_model(trainer, best_model)

        logging.info("Evaluating on test set...")
        pandas_prediction_callback = PandasPredictionCallback(
            top_k=max(self.config["metrics"]["ks"]),
            query_column="user_id",
            item_column="item_id",
            rating_column="score",
            postprocessors=[RemoveSeenItems(self.seq_test_dataset)],
        )
        L.Trainer(callbacks=[pandas_prediction_callback], inference_mode=True).predict(
            best_model, dataloaders=prediction_dataloader, return_predictions=False
        )

        result = pandas_prediction_callback.get_result()
        recommendations = self.tokenizer.query_and_item_id_encoder.inverse_transform(
            result
        )
        test_metrics = self.calculate_metrics(recommendations, self.raw_test_gt)
        logging.info(test_metrics)
        test_metrics.to_csv(
            os.path.join(
                self.config["paths"]["results_dir"],
                f"{self.model_save_name}_{self.dataset_name}_test_metrics.csv",
            ),
        )

