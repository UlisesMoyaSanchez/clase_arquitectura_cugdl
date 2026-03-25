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
from lightning.pytorch.cli import LightningCLI
from model import DogCatModule
from dataset import DogCatDataModule


def main():
    torch.set_float32_matmul_precision("high")  # Usa Tensor Cores en GPU Ampere+
    LightningCLI(DogCatModule, DogCatDataModule)


if __name__ == "__main__":
    main()
