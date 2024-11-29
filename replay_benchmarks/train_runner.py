import logging
import os

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from replay_benchmarks.base_runner import BaseRunner
from replay.metrics import OfflineMetrics, Recall, Precision, MAP, NDCG, HitRate, MRR
from replay.metrics.torch_metrics_builder import metrics_to_df
from replay.models.nn.sequential import SasRec, Bert4Rec
from replay.models.nn.optimizer_utils import FatOptimizerFactory
from replay.models.nn.sequential.callbacks import ValidationMetricsCallback
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
        self.logger = CSVLogger(
            save_dir=config["paths"]["log_dir"], name=self.model_name
        )

    def initialize_model(self):
        """Initialize the model based on the configuration."""
        model_config = {
            "tensor_schema": self.tensor_schema,
            "optimizer_factory": FatOptimizerFactory(
                learning_rate=self.model_cfg["training_params"]["learning_rate"]
            ),
        }
        model_config.update(self.model_cfg["model_params"])

        if self.model_name.lower() == "sasrec":
            return SasRec(**model_config)
        elif self.model_name.lower() == "bert4rec":
            return Bert4Rec(**model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

    def prepare_dataloaders(
        self,
        seq_train_dataset,
        seq_validation_dataset,
        seq_validation_gt,
        seq_test_dataset,
    ):
        """Initialize dataloaders for training, validation, and testing."""
        logging.info("Preparing dataloaders objects...")
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

        if self.model_name.lower() in dataset_mapping:
            TrainingDataset, ValidationDataset, PredictionDataset = dataset_mapping[
                self.model_name.lower()
            ]
        else:
            raise ValueError(
                f"Unsupported model type for dataloaders: {self.model_name}"
            )

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
        init_args = {"query_column": "user_id", "rating_column": "score"}

        metrics = OfflineMetrics(metrics_list, **init_args)(predictions, ground_truth)
        return metrics_to_df(metrics)

    def save_model(self, model, checkpoint_callback):
        """Save the best model checkpoint to the specified directory."""
        best_model_path = checkpoint_callback.best_model_path
        save_path = self.config["paths"]["model_dir"]
        model.load_from_checkpoint(best_model_path).save(
            f"{save_path}/{self.model_name}_best.pt"
        )
        logging.info(f"Best model saved at: {save_path}/{self.model_name}_best.pt")

    def run(self):
        """Execute the training pipeline."""
        logging.info("Preparing datasets for training.")
        train_events, validation_events, validation_gt, test_events, test_gt = (
            self.load_data()
        )

        train_dataset, val_dataset, val_gt_dataset, test_dataset, test_gt_dataset = (
            self.prepare_datasets(
                train_events, validation_events, validation_gt, test_events, test_gt
            )
        )

        (
            seq_train_dataset,
            seq_validation_dataset,
            seq_validation_gt,
            seq_test_dataset,
        ) = self.prepare_seq_datasets(
            train_dataset, val_dataset, val_gt_dataset, test_dataset, test_gt_dataset
        )

        train_dataloader, val_dataloader, prediction_dataloader = (
            self.prepare_dataloaders(
                seq_train_dataset,
                seq_validation_dataset,
                seq_validation_gt,
                seq_test_dataset,
            )
        )

        model = self.initialize_model()
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config["paths"]["checkpoint_dir"],
            save_top_k=1,
            verbose=True,
            monitor="recall@10",
            mode="max",
        )

        validation_metrics_callback = ValidationMetricsCallback(
            metrics=self.config["metrics"]["types"],
            ks=self.config["metrics"]["ks"],
            item_count=train_dataset.item_count,
            postprocessors=[RemoveSeenItems(seq_validation_dataset)],
        )

        trainer = L.Trainer(
            max_epochs=self.model_cfg["training_params"]["max_epochs"],
            callbacks=[checkpoint_callback, validation_metrics_callback],
            logger=self.logger,
        )

        logging.info("Starting model training.")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
        ) as prof:
            trainer.fit(model, train_dataloader, val_dataloader)

        prof.export_chrome_trace(
            os.path.join(
                self.config["paths"]["log_dir"],
                f"{self.model_name}_{self.dataset_name}_profile.json",
            )
        )
        logging.info("Training completed.")
