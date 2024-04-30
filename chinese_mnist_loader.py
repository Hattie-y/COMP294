import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def load_chinese_mnist(base_path: Path = Path("chinese_mnist")):
    csv = pd.read_csv(base_path / 'chinese_mnist.csv')
    images = []
    labels = []
    for idx, row in csv.iterrows():
        image_path = base_path / 'data' / 'data' / f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg"
        with Image.open(image_path) as img:
            image_tensor = transforms.PILToTensor()(img)
            images.append(image_tensor.unsqueeze(0))  # 保证图像张量是四维的
            labels.append(int(row['code']) - 1)  # 从CSV的code列减1, 确保为整数

    images_tensor = torch.stack(images).squeeze(1)  # 如果不需要额外的维度
    labels_tensor = torch.tensor(labels)

    print(images_tensor.shape, labels_tensor.shape)  # 打印出形状以确认

    return images_tensor, labels_tensor
