from typing import Any, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl


class MLGraphModule(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        graph: nn.Module,
        num_labels: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["backbone", "graph"], logger=False)

        # model
        self.graph = graph
        self.image_representor = backbone.get_backbone()

        # loss function
        self.criterion = nn.MultiLabelSoftMarginLoss()

        # metrics
        self.init_metrics(num_labels)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def init_metrics(self, num_labels: int):
        self.train_acc = Accuracy(task="multilabel", num_labels=num_labels)
        self.val_acc = Accuracy(task="multilabel", num_labels=num_labels)
        self.test_acc = Accuracy(task="multilabel", num_labels=num_labels)

        self.train_f1 = F1Score(task="multilabel", num_labels=num_labels)
        self.val_f1 = F1Score(task="multilabel", num_labels=num_labels)
        self.test_f1 = F1Score(task="multilabel", num_labels=num_labels)

    def forward(
        self,
        img: torch.Tensor,
        label_embedding: torch.Tensor,
        adj: torch.Tensor,
    ):
        # image representation
        features = self.image_representor(img)
        features = F.max_pool2d(features, features.shape[2:]).squeeze(2)

        # label embedding
        label_embedding = self.graph(label_embedding, adj)

        # combine
        output = torch.matmul(label_embedding, features)
        return output

    def step(self, batch: Any):
        (img, label_embedding, adj), target = batch

        output = self.forward(img, label_embedding, adj).squeeze()
        loss = self.criterion(output, target)
        preds = torch.sigmoid(output) > 0.5
        return loss, preds, target

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update metrics
        self.val_loss.update(loss)
        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)

        # log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)

        # Log the best accuracy seen so far
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        # Reset metrics at epoch end (optional but recommended)
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}
