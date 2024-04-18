# from dataset_split import get_dataloaders
# from model import FusionModelWithAttention
# import torch.nn as nn
# import torch
# import matplotlib.pyplot as plt
# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
# # 设置基本目录和批次大小
# base_output_dir = 'S:/Dissertation-dataset/example2'
# batch_size = 64
#
# train_losses = []
# train_accuracies = []
# val_losses = []
# val_accuracies = []
#
# # 获取数据加载器
# loaders = get_dataloaders(base_output_dir, batch_size)
#
# num_keypoints = 25 * 3  # 有25个关键点，每个关键点有x, y, 和置信度，总共75个值
# num_classes = 4  # 有4个类别，'normal', 'confusion', 'fall_dizzy', 'fall_weakness'
# model = FusionModelWithAttention(num_keypoints, num_classes)
#
# # 检查 CUDA 是否可用，并据此设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # 将模型移动到设定的设备
# model.to(device)
#
# criterion = nn.CrossEntropyLoss()  # 对于分类任务常用的损失函数
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
#
#
# def main():
#     num_epochs = 64
#     best_val_accuracy = 0.0  # 用于保存最好的验证准确率
#     model_save_path = 'S:/Dissertation-dataset/model.pth'  # 模型保存路径
#     for epoch in range(num_epochs):
#         model.train()  # 将模型设置为训练模式
#         total_train_loss = 0
#         total_train_correct = 0
#         total_train_samples = 0
#
#         for category in ['normal', 'confusion', 'fall_dizzy', 'fall_weakness']:
#             for images, keypoints, labels in loaders[category]['train']:
#                 images = images.to(device)
#                 keypoints = keypoints.view(keypoints.size(0), -1).to(device)
#                 labels = labels.to(device)
#
#                 output = model(images, keypoints)
#                 loss = criterion(output, labels)
#                 total_train_loss += loss.item()
#                 _, predicted = torch.max(output, 1)
#                 total_train_correct += (predicted == labels).sum().item()
#                 total_train_samples += labels.size(0)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#         avg_train_loss = total_train_loss / total_train_samples
#         train_accuracy = total_train_correct / total_train_samples
#
#         # 验证阶段
#         model.eval()  # 将模型设置为评估模式
#         total_val_loss = 0
#         total_val_correct = 0
#         total_val_samples = 0
#
#         for category in ['normal', 'confusion', 'fall_dizzy', 'fall_weakness']:
#             with torch.no_grad():
#                 for images, keypoints, labels in loaders[category]['valid']:
#                     images = images.to(device)
#                     keypoints = keypoints.view(keypoints.size(0), -1).to(device)
#                     labels = labels.to(device)
#
#                     output = model(images, keypoints)
#                     loss = criterion(output, labels)
#                     total_val_loss += loss.item()
#                     _, predicted = torch.max(output, 1)
#                     total_val_correct += (predicted == labels).sum().item()
#                     total_val_samples += labels.size(0)
#
#         avg_val_loss = total_val_loss / total_val_samples
#         val_accuracy = total_val_correct / total_val_samples
#
#         print(f'Epoch {epoch + 1}:')
#         print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
#         print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
#         # 在每个 epoch 结束后添加
#         train_losses.append(avg_train_loss)
#         train_accuracies.append(train_accuracy)
#         val_losses.append(avg_val_loss)
#         val_accuracies.append(val_accuracy)
#
#         # 如果当前模型的验证准确率是最好的，保存模型
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             torch.save(model.state_dict(), model_save_path)
#             print(f'Model saved to {model_save_path}')
#
#
# if __name__ == "__main__":
#     main()
#
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Loss Over Epochs')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Train Accuracy')
# plt.plot(val_accuracies, label='Validation Accuracy')
# plt.title('Accuracy Over Epochs')
# plt.legend()
#
# plt.show()


# 导入必要的库
from dataset_split import get_dataloaders_for_fold
from model import FusionModelWithAttention
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 设置基本参数
base_output_dir = 'S:/Dissertation-dataset/example2'
batch_size = 64
num_epochs = 64
k_folds = 5
num_keypoints = 25 * 3  # 假设每个关键点有x, y和置信度值
num_classes = 4  # 类别：'normal', 'confusion', 'fall_dizzy', 'fall_weakness'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# 初始化用于存储每一折平均损失和准确率的数组
fold_train_losses = []
fold_val_losses = []
fold_train_accuracies = []
fold_val_accuracies = []

# 确保保存模型的目录存在
model_save_dir = 'S:/Dissertation-dataset/models'
os.makedirs(model_save_dir, exist_ok=True)

# 对每一折进行训练和验证
for fold in range(k_folds):
    print(f"Starting fold {fold + 1}/{k_folds}")

    # 为当前折初始化用于存储每个epoch指标的数组
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 为每一折初始化模型
    model = FusionModelWithAttention(num_keypoints, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # 加载当前折的数据
    loaders = get_dataloaders_for_fold(base_output_dir, batch_size, k_folds, fold)
    best_val_accuracy = 0.0
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        # 训练循环
        for category in loaders.keys():
            for images, keypoints, labels in loaders[category]['train']:
                images, keypoints, labels = images.to(device), keypoints.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images, keypoints)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train_correct += (predicted == labels).sum().item()
                total_train_samples += labels.size(0)

        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = total_train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # 验证循环
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for category in loaders.keys():
                for images, keypoints, labels in loaders[category]['valid']:
                    images, keypoints, labels = images.to(device), keypoints.to(device), labels.to(device)
                    outputs = model(images, keypoints)
                    loss = criterion(outputs, labels)

                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val_correct += (predicted == labels).sum().item()
                    total_val_samples += labels.size(0)

        avg_val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # 输出训练和验证结果
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")

        # 检查当前epoch的验证准确率是否比之前记录的最佳准确率更高
        if val_accuracy > best_val_accuracy:
            # 如果是，更新最佳验证准确率
            best_val_accuracy = val_accuracy
            # 构建模型保存路径，包括折数和epoch
            model_save_path = os.path.join(model_save_dir, f'model_fold_{fold + 1}_best.pth')
            # 保存模型
            torch.save(model.state_dict(), model_save_path)
            # 打印模型已保存的消息
            print(f'Saved best model of fold {fold + 1} to {model_save_path}')

        # 存储每一折的平均训练和验证损失以及准确率
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_train_accuracies.append(train_accuracies)
        fold_val_accuracies.append(val_accuracies)

    # 绘制当前折的训练和验证损失图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.title(f'Loss over Epochs (Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制当前折的训练和验证准确率图
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracy over Epochs (Fold {fold + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 显示图表
    plt.show()
