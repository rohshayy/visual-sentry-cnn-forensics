import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from visual_sentry_model import VisualSentry

def train_engine():
    # Production Touch: Data Augmentation + Size Matching for Full Words
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),                 # Upscale to accommodate entire words
        transforms.RandomRotation(degrees=10),        # Simulates real paper alignment issues
        transforms.RandomInvert(p=0.5),              # Teaches AI to read dark ink on white backgrounds
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading EMNIST Forensic Set with Adaptive Normalization...")
    train_set = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = VisualSentry(num_classes=47)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    print("\nTraining Visual-Sentry Spatial Engine (10 Epochs)...")

    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/10 - Spatial Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "visualsentry_v1.pth")
    print("\nModel saved cleanly as 'visualsentry_v1.pth'.")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='darkblue', linewidth=2)
    plt.title("Visual-Sentry Gradient Convergence Map")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_engine()