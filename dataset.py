import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import config

class DogCatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directorio con subcarpetas 'dogs' y 'cats'
            transform: Transformaciones a aplicar
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['dogs', 'cats']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(phase='train'):
    """Devuelve las transformaciones según config"""
    if phase == 'train':
        t = config.TRAIN_TRANSFORMS
        transform_list = [
            transforms.Resize(t['resize']),
            transforms.RandomHorizontalFlip(p=0.5) if t['random_horizontal_flip'] else None,
            transforms.RandomRotation(t['random_rotation']) if t['random_rotation'] else None,
            transforms.ToTensor(),
        ]
        if t['normalize']:
            transform_list.append(transforms.Normalize(mean=t['mean'], std=t['std']))
    else:
        t = config.VAL_TRANSFORMS
        transform_list = [
            transforms.Resize(t['resize']),
            transforms.ToTensor(),
        ]
        if t['normalize']:
            transform_list.append(transforms.Normalize(mean=t['mean'], std=t['std']))
    
    # Filtrar None
    transform_list = [tr for tr in transform_list if tr is not None]
    return transforms.Compose(transform_list)
