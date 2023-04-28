import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob


from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data_dir, min_side=512):
        self.train_list = []
        files = glob(f"{data_dir}/*.jpg")
        for line in files:
            self.train_list.append(line)
        self.min_side = min_side

    def __getitem__(self, index):
        img_name = self.train_list[index]
        label_name = img_name[:-4] + ".png"
        img = cv2.imread(img_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        # Resize image using PIL
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = Image.fromarray(label)
        img = img.resize((512, 512), resample=Image.BILINEAR)
        label = label.resize((512, 512), resample=Image.NEAREST)

        # Convert image to numpy array and normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.clip(img, -1, 1)

        # Convert label to tensor
        label = torch.from_numpy(np.array(label)).long()

        with Image.open(img_path).convert('RGB') as img:
            img_tensor = torch.Tensor(img)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 将维度转变为 [1, C, H, W]

        label = self.labels[index]

        return img_tensor, label

    def __len__(self):
        return len(self.train_list)
