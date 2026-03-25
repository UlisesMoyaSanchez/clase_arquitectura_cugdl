import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import config
import torchvision
from dataset import DogCatDataset, get_transforms
from model import SimpleCNN
from utils import calculate_accuracy, set_seed
import matplotlib.pyplot as plt

def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [Train]")):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_acc += calculate_accuracy(outputs, labels) * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_acc += calculate_accuracy(outputs, labels) * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def run_training():
    set_seed(42)
    
    # Crear directorios si no existen
    os.makedirs("models", exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Preparar datasets
    full_dataset = DogCatDataset(
        root_dir=config.TRAIN_DIR,
        transform=get_transforms('train')
    )
    
    # Dividir en train (80%) y val (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Aplicar transformaciones diferentes a validación (sin augmentation)
    val_dataset.dataset.transform = get_transforms('val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Modelo, criterio, optimizador
    model = SimpleCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR)
    
    # Guardar imágenes de ejemplo en TensorBoard (opcional)
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images[:16], normalize=True)
    writer.add_image('Ejemplos de entrenamiento', img_grid, 0)
    
    # Listas para almacenar métricas para plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Entrenamiento
    best_val_acc = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE, writer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE, writer, epoch)
        
        # Almacenar métricas para plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Mejor modelo guardado con accuracy {val_acc:.4f}")
    
    writer.close()
    
    # Plotting de curvas
    epochs = list(range(1, config.EPOCHS + 1))
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accs, label='Val Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Entrenamiento completado. Curvas guardadas en 'training_curves.png'.")
