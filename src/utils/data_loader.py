from torch.utils.data import Dataset
import pandas as pd
import cv2
import torch
import numpy as np

class GAN_data_loader(Dataset):
    def __init__(self, csv_path, dataset_type = "train", transform = None, device = "cpu"):
        csv_df = pd.read_csv(csv_path)
        self.dataset_type = "train"
        self.images_path = np.asarray(csv_df.loc[csv_df.datasets == self.dataset_type, ["images_path"]]).squeeze()
        self.class_id = np.asarray(csv_df.loc[csv_df.datasets == self.dataset_type, ["class_id"]])
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR) / 255.0
        image = torch.Tensor(np.array(image.transpose([2, 0, 1]))).to(self.device)

        if self.transform:
            image = self.transform(image)

        return image, 0