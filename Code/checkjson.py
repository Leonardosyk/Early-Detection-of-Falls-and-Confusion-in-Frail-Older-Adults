import os
import json

# json_dir = 'S:\\Dissertation-dataset\\example2\\keypoints'  # 您的关键点数据目录
#
# def check_keypoints_shape(json_dir):
#     for folder_name in ['normal', 'confusion', 'fall_dizzy', 'fall_weakness']:
#         folder_path = os.path.join(json_dir, folder_name)
#         for json_file in os.listdir(folder_path):
#             if json_file.endswith('.json'):
#                 json_path = os.path.join(folder_path, json_file)
#                 with open(json_path, 'r') as f:
#                     keypoints = json.load(f)
#                     print(f"{json_file}: {len(keypoints)} keypoints, shape: {len(keypoints[0]) if keypoints else 'N/A'}")
#
# check_keypoints_shape(json_dir)


import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
