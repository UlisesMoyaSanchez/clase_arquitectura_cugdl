import torch

# Rutas
DATA_DIR = "data"
TRAIN_DIR = f"{DATA_DIR}"  # Asumimos que dentro hay subcarpetas 'dogs' y 'cats'
MODEL_SAVE_PATH = "models/best_model.pth"
LOGS_DIR = "runs"

# Hiperparámetros
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
IMG_SIZE = 224  # Tamaño de imagen para resnet (224x224)
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuración de transforms (data augmentation)
TRAIN_TRANSFORMS = {
    "resize": (IMG_SIZE, IMG_SIZE),
    "random_horizontal_flip": True,
    "random_rotation": 10,
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],  # Media de ImageNet
    "std": [0.229, 0.224, 0.225]
}

VAL_TRANSFORMS = {
    "resize": (IMG_SIZE, IMG_SIZE),
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}
