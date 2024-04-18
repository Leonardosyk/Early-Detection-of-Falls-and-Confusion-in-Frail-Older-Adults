import cv2
import os


# Function to split a video into specified segments
def split_video(video_path, start_normal, duration_normal, end_condition, output_normal_dir, output_condition_dir,
                condition_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame numbers for the split
    end_frame_normal = int(fps * start_normal + fps * duration_normal)
    start_frame_condition = int(total_frames - fps * end_condition)

    # Prepare video writer for the normal part
    normal_video_path = os.path.join(output_normal_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    normal_writer = cv2.VideoWriter(normal_video_path, fourcc, fps,
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Prepare video writer for the condition part
    condition_video_path = os.path.join(output_condition_dir, condition_name + '_' + os.path.basename(video_path))
    condition_writer = cv2.VideoWriter(condition_video_path, fourcc, fps,
                                       (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame < end_frame_normal:
            normal_writer.write(frame)
        elif current_frame >= start_frame_condition:
            condition_writer.write(frame)

        current_frame += 1

    cap.release()
    normal_writer.release()
    condition_writer.release()


# Paths configuration
base_input_dir = 'S:/Dissertation-dataset/example/video/video'  # Update with the correct path to the input videos
output_dirs = {
    'normal': 'S:/Dissertation-dataset/normal',  # Update with the correct path to the 'normal' output folder
    'confusion': 'S:/Dissertation-dataset/confusion',  # Update with the correct path to the 'confusion' output folder
    'fall_dizzy': 'S:/Dissertation-dataset/fall_dizzy',  # Update with the correct path to the 'fall_dizzy' output folder
    'fall_weakness': 'S:/Dissertation-dataset/fall_weakness',  # Update with the correct path to the 'fall_weakness' output folder
}

# Make sure all the output directories exist
for output_dir in output_dirs.values():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Define the folders and corresponding condition names
video_folders = {
    'confusion_gait': 'confusion',
    'confusion_lost': 'confusion',
    'fall_dizzy_forward': 'fall_dizzy',
    'fall_dizzy_side': 'fall_dizzy',
    'fall_weakness_forward': 'fall_weakness',
    'fall_weakness_side': 'fall_weakness',
}
for folder, condition in video_folders.items():
    input_folder_path = os.path.join(base_input_dir, folder)
    # Check if the input folder exists
    if not os.path.exists(input_folder_path):
        print(f"Input folder {input_folder_path} does not exist. Skipping...")
        continue

    # List all video files in the folder
    video_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.mp4', '.avi'))]

    # Process each video
    for video_file in video_files:
        video_path = os.path.join(input_folder_path, video_file)
        print(f"Processing {video_file}...")

        # Call the function to split the video
        split_video(
            video_path=video_path,
            start_normal=0,  # Start of the video
            duration_normal=6,  # Duration of the normal part
            end_condition=4,  # Duration of the condition part from the end
            output_normal_dir=output_dirs['normal'],
            output_condition_dir=output_dirs[condition],
            condition_name=condition
        )
print("Video processing completed.")