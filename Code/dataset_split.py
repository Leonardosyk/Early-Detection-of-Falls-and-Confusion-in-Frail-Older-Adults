import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import json

base_output_dir = 'S:\Dissertation-dataset\example2'


class ImageKeypointsDataset(Dataset):
    def __init__(self, img_dir, json_dir, transform=None):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.labels = {  # 创建标签字典
            'normal': 0,
            'confusion': 1,
            'fall_dizzy': 2,
            'fall_weakness': 3
        }

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file_name)
        img = Image.open(img_path).convert('RGB')

        img_number = img_file_name.split('_')[1].split('.')[0]
        json_file_name = f"keypoints_{img_number}.json"
        json_path = os.path.join(self.json_dir, json_file_name)

        with open(json_path, 'r') as f:
            keypoints = json.load(f)

        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).squeeze(0)  # 移除多余的批次维度，如果有的话

        if self.transform:
            img = self.transform(img)

        label = self.labels[os.path.basename(self.img_dir)]

        return img, keypoints_tensor, label


def custom_collate_fn(batch):
    processed_batch = []
    for item in batch:
        image, keypoints, label = item
        # 检查并调整关键点的形状
        if keypoints.dim() == 3 and keypoints.size(0) == 1:
            keypoints = keypoints.squeeze(0)
        elif keypoints.dim() == 2 and keypoints.size(0) == 75:
            keypoints = keypoints.view(25, 3)
        elif keypoints.dim() != 2 or keypoints.size(0) != 25:
            # 如果形状不符，跳过这个样本
            continue
        processed_batch.append((image, keypoints, label))

    return torch.utils.data.dataloader.default_collate(processed_batch)




from sklearn.model_selection import KFold

def get_dataloaders_for_fold(base_output_dir, batch_size, n_splits, fold_index):
    # 定义转换
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loaders = {}

    # 使用KFold
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # 遍历每个类别的文件夹
    for folder in ['normal', 'confusion', 'fall_dizzy', 'fall_weakness']:
        img_dir = os.path.join(base_output_dir, "images", folder)
        json_dir = os.path.join(base_output_dir, "keypoints", folder)

        # 实例化数据集
        dataset = ImageKeypointsDataset(img_dir, json_dir, transform=train_transform)

        # 分割数据集
        for fold, (train_val_ids, test_ids) in enumerate(kfold.split(dataset)):
            if fold == fold_index:
                # 进一步分割训练和验证集
                split = int(len(train_val_ids) * 0.8)  # 假设我们用80%作为训练集，剩下的作为验证集
                train_ids = train_val_ids[:split]
                val_ids = train_val_ids[split:]

                train_subset = torch.utils.data.Subset(dataset, train_ids)
                val_subset = torch.utils.data.Subset(dataset, val_ids)
                test_subset = torch.utils.data.Subset(dataset, test_ids)

                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                          collate_fn=custom_collate_fn)
                valid_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                          collate_fn=custom_collate_fn)
                test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                         collate_fn=custom_collate_fn)

                # 保存加载器到字典
                loaders = {
                    'train': train_loader,
                    'valid': valid_loader,
                    'test': test_loader
                }

                break  # 已找到当前fold，不再继续

    return loaders


# def get_dataloaders(base_output_dir, batch_size):
#     # 为训练数据定义额外的数据增强操作
#     train_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomHorizontalFlip(),  # 随机水平翻转
#         transforms.RandomRotation(10),  # 随机旋转
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     # 验证和测试数据集使用的转换不包括数据增强
#     valid_test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     loaders = {}
#     split_ratios = [0.6, 0.2, 0.2]
#
#     for folder in ['normal', 'confusion', 'fall_dizzy', 'fall_weakness']:
#         img_dir = os.path.join(base_output_dir, "images", folder)
#         json_dir = os.path.join(base_output_dir, "keypoints", folder)
#
#         # 应用不同的转换
#         train_dataset = ImageKeypointsDataset(img_dir, json_dir, transform=train_transform)
#         valid_test_dataset = ImageKeypointsDataset(img_dir, json_dir, transform=valid_test_transform)
#
#         lengths = [int(len(train_dataset) * split) for split in split_ratios]
#         lengths[-1] = len(train_dataset) - sum(lengths[:-1])
#         train_set, valid_set, test_set = random_split(valid_test_dataset, lengths)
#
#         loaders[folder] = {
#             'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn),
#             'valid': DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn),
#             'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
#         }
#
#     return loaders