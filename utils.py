import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class RockDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx]['label'])  # 정수형 클래스

        if self.transform:
            image = self.transform(image)

        return image, label