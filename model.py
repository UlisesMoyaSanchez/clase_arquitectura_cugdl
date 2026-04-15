import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torchvision import models


# ---------------------------------------------------------------------------
# Arquitecturas backbone
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """CNN sencilla: 3 bloques Conv+ReLU+MaxPool seguidos de 2 capas FC."""

    def __init__(self, num_classes: int = 2, img_size: int = 224):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        # tres poolings: img_size → img_size/2 → img_size/4 → img_size/8
        feat_size = img_size // 8
        self.fc1     = nn.Linear(128 * feat_size * feat_size, 256)
        self.fc2     = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def _build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class DogCatModule(L.LightningModule):
    """LightningModule para clasificación binaria perros/gatos.

    Args:
        num_classes: Número de clases (por defecto 2).
        lr:          Learning rate del optimizador SGD.
        momentum:    Momentum del optimizador SGD.
        backbone:    Arquitectura a usar: ``'simplecnn'`` o ``'resnet18'``.
        pretrained:  Si ``backbone='resnet18'``, carga pesos ImageNet.
    """

    def __init__(
        self,
        num_classes: int = 2,
        lr: float = 1e-3,
        momentum: float = 0.9,
        backbone: str = "simplecnn",
        pretrained: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters()

        if backbone == "simplecnn":
            self.net = SimpleCNN(num_classes=num_classes, img_size=img_size)
        elif backbone == "resnet18":
            self.net = _build_resnet18(num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"backbone desconocido: '{backbone}'. Usa 'simplecnn' o 'resnet18'.")

        self.criterion = nn.CrossEntropyLoss()

        metric_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.val_acc   = torchmetrics.Accuracy(**metric_kwargs)
        self.test_acc  = torchmetrics.Accuracy(**metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train/loss", loss,           on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log("val/loss", loss,         on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc,  on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.log("test/loss", loss,          on_step=False, on_epoch=True)
        self.log("test/acc",  self.test_acc,  on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
        )

