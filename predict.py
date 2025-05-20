import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from models.model import get_model

# ✅ 클래스 이름: 학습에 사용된 순서와 일치해야 함
class_names = [
    'Andesite',
    'Basalt',
    'Etc',
    'Gneiss',
    'Granite',
    'Mud_Sandstone',
    'Weathered_Rock'
]

# ✅ 테스트 이미지 로더
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
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

# ✅ 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 데이터 로딩
test_dir = 'data/test'
test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ 모델 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(class_names)
model = get_model(num_classes)
model.load_state_dict(torch.load('rock_model.pth', map_location=device))
model.to(device)
model.eval()

# ✅ 예측 및 결과 저장
results = []

with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()

        for fname, pred in zip(filenames, preds):
            image_id = os.path.splitext(fname)[0]  # 'TEST_00001.jpg' → 'TEST_00001'
            rock_type = class_names[pred]
            results.append([image_id, rock_type])

# ✅ DataFrame으로 저장
submission = pd.DataFrame(results, columns=['ID', 'rock_type'])
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv 생성 완료!")
