#!/usr/bin/env python3
"""
Quick validation script for FALCON optimizer
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.cifar_vgg import vgg11_bn
from optim.falcon_v5 import FALCONv5
import time

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=2
    )
    
    # Model
    model = vgg11_bn().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = FALCONv5(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        weight_decay=5e-4,
        retain_energy_start=0.99,
        retain_energy_end=0.50,
        falcon_every_start=2,
        falcon_every_end=1,
        apply_stages="2,3,4",
        mask_interval=10,
        ema_decay=0.999,
        muon_lr_mult=1.25
    )
    
    print("\n" + "="*60)
    print("FALCON v5 Quick Validation")
    print("="*60)
    print(f"Model: VGG11-BN")
    print(f"Dataset: CIFAR-10 Test Set")
    print(f"Batch Size: 512")
    print(f"Optimizer: FALCONv5")
    print("="*60 + "\n")
    
    # Train for 1 epoch
    model.train()
    train_loss = 0
    start_time = time.time()
    
    # Get a small subset for quick validation
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    )
    
    # Use only first 10% for quick validation
    subset_size = len(trainset) // 10
    trainset = torch.utils.data.Subset(trainset, range(subset_size))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=512, shuffle=True, num_workers=2
    )
    
    print("Training for 1 epoch (10% data)...")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(trainloader)}: Loss = {loss.item():.4f}")
    
    epoch_time = time.time() - start_time
    avg_loss = train_loss / len(trainloader)
    
    print(f"\nTraining completed in {epoch_time:.2f}s")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # Test
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    
    print("\n" + "="*60)
    print("✓ FALCON v5 validation completed successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
