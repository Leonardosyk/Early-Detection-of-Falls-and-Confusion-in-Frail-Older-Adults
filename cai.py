#/* --- Coding:utf-8 --- */
"""
author:Flower_River
time:2022.08.19
"""
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('os.path.dirname(os.path.realpath(__file__)) is : ', dir_path)
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # 修改参数
    params["net_resolution"] = "368x368"  # 修改分辨率，可以降低对显存的占用

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    video = cv2.VideoCapture("S:/Dissertation-dataset/example/video/video/confusion_gait/CGFM1.mp4")


    if video.isOpened():
        # video.read() 一帧一帧地读取
        # open 得到的是一个布尔值，就是 True 或者 False
        # frame 得到当前这一帧的图像
        open, frame = video.read()
    else:
        open = False


    while open:

        ret, frame = video.read()
        # 如果读到的帧数不为空，那么就继续读取，如果为空，就退出

        if frame is None:
            break
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        if ret == True:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData[:])

            # 这里使用 waitKey 可以控制视频的播放速度，数值越小，播放速度越快
            # 这里等于 27 也即是说按下 ESC 键即可退出该窗口
            if cv2.waitKey(10) & 0xFF == 27:
                break
    video.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(e)
    sys.exit(-1)
