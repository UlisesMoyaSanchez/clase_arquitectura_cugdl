import os
from typing import List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import lightning as L


class DogCatDataset(Dataset):
    """Dataset de perros y gatos. Lee imágenes de subdirectorios 'dogs' y 'cats'."""

    CLASSES = ['dogs', 'cats']

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        self.images: List[str] = []
        self.labels: List[int] = []

        for cls in self.CLASSES:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(
    img_size: int,
    mean: List[float],
    std: List[float],
    augment: bool = False,
) -> transforms.Compose:
    """Construye el pipeline de transformaciones."""
    steps = [transforms.Resize((img_size, img_size))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(steps)


class DogCatDataModule(L.LightningDataModule):
    """LightningDataModule con split train / val / test.

    Args:
        data_dir:     Directorio raíz con subcarpetas 'dogs' y 'cats'.
        batch_size:   Tamaño de lote.
        img_size:     Lado de la imagen cuadrada redimensionada.
        train_ratio:  Fracción para entrenamiento (p.ej. 0.70).
        val_ratio:    Fracción para validación   (p.ej. 0.15).
                      El resto (1 - train - val) se usa como test.
        mean:         Media de normalización por canal.
        std:          Desviación estándar de normalización por canal.
        num_workers:  Hilos para DataLoader.
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        img_size: int = 224,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        mean: List[float] = (0.485, 0.456, 0.406),
        std: List[float] = (0.229, 0.224, 0.225),
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert train_ratio + val_ratio < 1.0, (
            "train_ratio + val_ratio debe ser menor que 1.0"
        )

    def setup(self, stage: Optional[str] = None):
        full = DogCatDataset(self.hparams.data_dir)
        n = len(full)
        n_train = int(n * self.hparams.train_ratio)
        n_val = int(n * self.hparams.val_ratio)
        n_test = n - n_train - n_val

        train_ds, val_ds, test_ds = random_split(
            full,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        # Aplicar transforms independientes por split
        train_transform = build_transforms(
            self.hparams.img_size, self.hparams.mean, self.hparams.std, augment=True
        )
        eval_transform = build_transforms(
            self.hparams.img_size, self.hparams.mean, self.hparams.std, augment=False
        )

        train_ds.dataset = _TransformedSubset(train_ds.dataset, train_ds.indices, train_transform)
        val_ds.dataset   = _TransformedSubset(val_ds.dataset,   val_ds.indices,   eval_transform)
        test_ds.dataset  = _TransformedSubset(test_ds.dataset,  test_ds.indices,  eval_transform)

        # Reasignar índices a 0..N para el wrapper
        train_ds.indices = list(range(len(train_ds.indices)))
        val_ds.indices   = list(range(len(val_ds.indices)))
        test_ds.indices  = list(range(len(test_ds.indices)))

        self.train_ds = train_ds
        self.val_ds   = val_ds
        self.test_ds  = test_ds

        print(
            f"Dataset split — train: {len(self.train_ds)} | "
            f"val: {len(self.val_ds)} | test: {len(self.test_ds)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


class _TransformedSubset(Dataset):
    """Wrapper que aplica un transform propio a un subconjunto de un Dataset base."""

    def __init__(self, base: DogCatDataset, indices: List[int], transform):
        self.images = [base.images[i] for i in indices]
        self.labels = [base.labels[i] for i in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert('RGB')
        return self.transform(image), self.labels[idx]
