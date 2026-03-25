import torch
import numpy as np
import random

def set_seed(seed=42):
    """Fijar semillas para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_accuracy(outputs, labels):
    """Calcula precisión (accuracy)"""
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)
