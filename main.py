"""Punto de entrada principal — usa LightningCLI.

Ejemplos de uso:
    # Entrenamiento
    python main.py fit --config configs/fit.yaml

    # Validación (requiere ckpt_path en el YAML o como argumento)
    python main.py validate --config configs/val.yaml

    # Test
    python main.py test --config configs/val.yaml --ckpt_path models/best_model.ckpt
"""

import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from model import DogCatModule
from dataset import DogCatDataModule


class _OptionalSaveConfigCallback(SaveConfigCallback):
    """Skip saving config when the logger has no log_dir (e.g. MLFlowLogger)."""

    def setup(self, trainer, pl_module, stage=None):
        if trainer.logger and getattr(trainer.logger, "log_dir", None) is None:
            return
        super().setup(trainer, pl_module, stage=stage)


def main():
    torch.set_float32_matmul_precision("high")  # Usa Tensor Cores en GPU Ampere+
    LightningCLI(
        DogCatModule,
        DogCatDataModule,
        save_config_callback=_OptionalSaveConfigCallback,
    )


if __name__ == "__main__":
    main()
