# from instantiated import model, loaders, criterion
# import torch
#
# model_save_path = 'S:/Dissertation-dataset/model.pth'  # 模型保存路径
# # Load the best saved model
# model.load_state_dict(torch.load(model_save_path))
# print("It's going to test module")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'using device', {device})
# # Test the model
# model.eval()  # Set the model to evaluation mode
# total_test_loss = 0
# total_test_correct = 0
# total_test_samples = 0
# # test_accuracies = []
# # test_losses = []
# with torch.no_grad():
#     for category in ['normal', 'confusion', 'fall_dizzy', 'fall_weakness']:
#         for images, keypoints, labels in loaders[category]['test']:
#             images = images.to(device)
#             keypoints = keypoints.view(keypoints.size(0), -1).to(device)
#             labels = labels.to(device)
#
#             # Forward pass
#             output = model(images, keypoints)
#             loss = criterion(output, labels)
#             total_test_loss += loss.item()
#
#             # Calculate the number of correct samples
#             _, predicted = torch.max(output, 1)  # just need the index of correct predicted
#             total_test_correct += (predicted == labels).sum().item()
#             total_test_samples += labels.size(0)
# avg_test_losses = total_test_loss / total_test_samples
# avg_test_accuracies = total_test_correct / total_test_samples
# print(f'Test loss: {avg_test_losses:.4f},Test accuracies: {avg_test_accuracies:.4f}')
# # test_accuracies.append(avg_test_accuracies)
# # test_losses.append(avg_test_losses)
#
# # import matplotlib.pyplot as plt
# # import os
# #
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# #
# # plt.figure(figsize=(10, 5))
# # # plt.subplot(1, 2, 1)
# # plt.plot(test_accuracies, label="test accuracies")
# # plt.plot(test_losses, label="test losses")
# # plt.title("Test accuracies and losses")
# # plt.legend()


from dataset_split import get_dataloaders_for_fold
from model import FusionModelWithAttention
import torch
import torch.nn as nn
import os

# Parameters
base_output_dir = 'S:/Dissertation-dataset/example2'
batch_size = 64
k_folds = 5
num_keypoints = 25 * 3  # Assuming 25 keypoints with x, y, and confidence values
num_classes = 4  # 'normal', 'confusion', 'fall_dizzy', 'fall_weakness'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Ensure the directory for saving models exists
model_save_dir = 'S:/Dissertation-dataset/models'

# Initialize arrays to store the test results
fold_test_losses = []
fold_test_accuracies = []

# Testing for each fold
for fold in range(k_folds):
    # Initialize the test metrics
    total_test_loss = 0
    total_test_correct = 0
    total_test_samples = 0

    # Load the best model from the current fold
    model_path = os.path.join(model_save_dir, f'model_fold_{fold + 1}_best.pth')
    model = FusionModelWithAttention(num_keypoints, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()  # Set the model to evaluation mode

    # Get the dataloaders for the current fold - ensure your function provides a test_loader
    loaders = get_dataloaders_for_fold(base_output_dir, batch_size, k_folds, fold)

    # Run the test loop
    with torch.no_grad():
        for images, keypoints, labels in loaders['test']:
            images = images.to(device)
            keypoints = keypoints.view(keypoints.size(0), -1).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, keypoints)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test_correct += (predicted == labels).sum().item()
            total_test_samples += labels.size(0)

    # Calculate the average loss and accuracy
    avg_test_loss = total_test_loss / total_test_samples
    avg_test_accuracy = total_test_correct / total_test_samples

    # Store the results
    fold_test_losses.append(avg_test_loss)
    fold_test_accuracies.append(avg_test_accuracy)

    # Print the results for the current fold
    print(f'Test results for fold {fold+1}: Loss = {avg_test_loss:.4f}, Accuracy = {avg_test_accuracy:.4f}')

# print or log the average test loss and accuracy across all folds
average_test_loss = sum(fold_test_losses) / k_folds
average_test_accuracy = sum(fold_test_accuracies) / k_folds
print(f'Average Test Loss: {average_test_loss:.4f}')
print(f'Average Test Accuracy: {average_test_accuracy:.4f}')
