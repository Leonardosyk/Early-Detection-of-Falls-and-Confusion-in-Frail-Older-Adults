import sys
import cv2
import os
import json
from sys import platform
import argparse

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('os.path.dirname(os.path.realpath(__file__)) is : ', dir_path)
    if platform == "win32":
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin'
        import pyopenpose as op
    else:
        sys.path.append('../../python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
args = parser.parse_known_args()

# Custom Params
params = dict()
params["model_folder"] = "../../../models/"
params["net_resolution"] = "368x368"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Video
video_path = "S:/Dissertation-dataset/example/video/video/confusion_gait/CGFM1.mp4"
video = cv2.VideoCapture(video_path)

# 基本目录配置
base_input_dir = "S:/Dissertation-dataset/example2/video"
base_output_dir = "S:/Dissertation-dataset/example2"


# 定义处理视频的函数
def process_video(input_video_path, output_img_dir, output_json_dir):
    video = cv2.VideoCapture(input_video_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # 保存处理后的帧为图像
        output_image_path = os.path.join(output_img_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_image_path, datum.cvOutputData)

        # 保存关键点为JSON
        if datum.poseKeypoints is not None:
            keypoints_data = datum.poseKeypoints.tolist()
            output_json_path = os.path.join(output_json_dir, f"keypoints_{frame_count:04d}.json")
            with open(output_json_path, 'w') as f:
                json.dump(keypoints_data, f)

        frame_count += 1

    video.release()


# 处理每个文件夹
folders = ['normal', 'confusion', 'fall_dizzy', 'fall_weakness']
for folder in folders:
    input_dir = os.path.join(base_input_dir, folder)
    output_img_dir = os.path.join(base_output_dir, "images", folder)
    output_json_dir = os.path.join(base_output_dir, "keypoints", folder)

    # 确保输出目录存在
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir)

    # 处理目录中的每个视频文件
    for video_file in os.listdir(input_dir):
        if video_file.endswith(('.mp4', '.avi')):
            video_path = os.path.join(input_dir, video_file)
            process_video(video_path, output_img_dir, output_json_dir)

print("Video processing completed.")