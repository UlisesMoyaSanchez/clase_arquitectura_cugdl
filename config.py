"""config.py — constantes legadas.

Con la migración a Lightning + LightningCLI todos los hiperparámetros
se gestionan en los YAMLs de configuración bajo configs/.

    configs/fit.yaml  →  entrenamiento
    configs/val.yaml  →  validación / test

Este módulo se conserva solo como referencia histórica.
"""
import torch

# Rutas
DATA_DIR         = "data"
TRAIN_DIR        = DATA_DIR
MODEL_SAVE_PATH  = "models/best_model.ckpt"
LOGS_DIR         = "runs"          # Directorio base; cada ejecución crea runs/v1/, runs/v2/, ...

# Hiperparámetros
BATCH_SIZE    = 32
EPOCHS        = 10
LEARNING_RATE = 0.001
MOMENTUM      = 0.9
IMG_SIZE      = 224
NUM_CLASSES   = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

