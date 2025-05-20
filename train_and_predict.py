import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from models.model import get_model
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, filename


def train_model(train_dir, batch_size=32, epochs=10, lr=0.0005):
    print("🚀 Training 시작")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = os.cpu_count() if os.name != 'nt' else 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\n📚 Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, labels in tqdm(train_loader, desc="🔄 Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"▶ Train Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="🔍 Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"✅ Val Accuracy: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names
            }, 'rock_model.pth')
            print("📦 Best model saved!")

    return class_names


def predict(test_dir, model_path='rock_model.pth'):
    print("\n🧪 Prediction 시작")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = os.cpu_count() if os.name != 'nt' else 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    model = get_model(len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="📸 Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            for fname, pred, conf in zip(filenames, preds.cpu(), confidences.cpu()):
                image_id = os.path.splitext(fname)[0]
                results.append([image_id, class_names[pred], f"{conf:.4f}"])

    submission = pd.DataFrame(results, columns=['ID', 'rock_type', 'confidence'])
    submission.to_csv('submission.csv', index=False)
    print("📄 submission.csv 저장 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()

    set_seed(42)

    class_list = train_model(args.train_dir, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    predict(args.test_dir)
