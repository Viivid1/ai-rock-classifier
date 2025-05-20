import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.model import get_model
from tqdm import tqdm

def main():
    # 하이퍼파라미터
    batch_size = 32
    epochs = 10
    lr = 0.0005
    train_dir = 'data/train'

    # 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # 데이터 로딩
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    print("클래스 목록:", dataset.classes)
    num_classes = len(dataset.classes)

    # train/val 분할
    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습 루프
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"▶ Train Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        # 검증
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"✅ Validation Accuracy: {val_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), 'rock_model.pth')
    print("✅ 모델 저장 완료")

if __name__ == "__main__":
    main()
