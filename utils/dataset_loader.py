import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(data_dir, batch_size=32, image_size=256, val_ratio=0.1, test_ratio=0.1, seed=42):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 정규화 추가!
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader, full_dataset.class_to_idx
