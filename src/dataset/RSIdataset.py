import glob
import json
import os
import random

import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader


class RSIdataset(Dataset):
    def __init__(self, root, index_list):
        super(RSIdataset, self).__init__()
        self.images_list = [
            [os.path.join(root, "images", str(i) + "_1.png"), os.path.join(root, "images", str(i) + "_2.png")]
            for i in index_list]
        self.gt_list = [
            [os.path.join(root, "gt", str(i) + "_1_label.png"), os.path.join(root, "gt", str(i) + "_2_label.png"),
             os.path.join(root, "gt", str(i) + "_change.png")] for i in index_list]

    def __getitem__(self, idx):
        image_data = self.images_list[idx]
        label_data = self.gt_list[idx]
        x1 = cv.imread(image_data[0])
        x1 = cv.cvtColor(x1, cv.COLOR_BGR2RGB)
        x1 = torch.tensor(x1, dtype=torch.float)
        x1 = x1.permute(2, 0, 1)

        x2 = cv.imread(image_data[1])
        x2 = cv.cvtColor(x2, cv.COLOR_BGR2RGB)
        x2 = torch.tensor(x2, dtype=torch.float)
        x2 = x2.permute(2, 0, 1)

        x1_label = cv.imread(label_data[0], flags=cv.IMREAD_GRAYSCALE)
        x1_label[x1_label <= 127] = 0
        x1_label[x1_label > 127] = 1
        x1_label = torch.unsqueeze(torch.tensor(x1_label), dim=0)
        x2_label = cv.imread(label_data[1], flags=cv.IMREAD_GRAYSCALE)
        x2_label[x2_label <= 127] = 0
        x2_label[x2_label > 127] = 1
        x2_label = torch.unsqueeze(torch.tensor(x2_label), dim=0)
        change = cv.imread(label_data[2], flags=cv.IMREAD_GRAYSCALE)
        change[change <= 127] = 0
        change[change > 127] = 1
        change = torch.unsqueeze(torch.tensor(change), dim=0)
        return x1, x2, x1_label, x2_label, change

    def __len__(self):
        return len(self.images_list)


def get_data(root=None, batch_size=1, num_workers=1, train_stage=True, split_train=0.8):
    if not os.path.exists(root):
        print("******数据路径错误，此文件夹不存在******")
    else:
        total_data = glob.glob(os.path.join(root, 'images', '*'))
        total_data = list(
            set(os.path.relpath(file_name, os.path.join(root, 'images')).split('_')[0] for file_name in total_data))
        if train_stage and split_train != 1:
            if os.path.exists("./data_split.json"):
                with open('./data_split.json', 'r') as f:
                    data_split = json.load(f)
                train, test = data_split["train"], data_split["test"]
            else:
                random.shuffle(total_data)
                train, test = total_data[0: int(split_train * len(total_data))], total_data[int(split_train * len(total_data)):]
                with open('./data_split.json', 'w') as f:
                    json.dump({"train": train, "test": test}, f)
            train = DataLoader(RSIdataset(root, train), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test = DataLoader(RSIdataset(root, test), batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return train, test
        elif train_stage:
            train = DataLoader(RSIdataset(root, total_data), batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
            return train
        else:
            test = DataLoader(RSIdataset(root, total_data), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
            return test


if __name__ == '__main__':
    root = r"E:\RSI-ChangeDetection\RSI-ChangeDetection\trainData"
    train = get_data(root=root, split_train=1, train_stage=False)
    for x1, x2, x1_label, x2_label, change in train:
        x1_label = torch.squeeze(x1_label)
        print(x1_label.shape)
        print(x1.shape)
        break
