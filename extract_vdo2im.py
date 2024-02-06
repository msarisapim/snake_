import cv2
import os
import pathlib

def extract_frames(video_path, output_folder, num_frames):
    # Extract video filename without extension
    video_name = pathlib.Path(video_path).stem

    # Create the output subfolders for this video if they don't exist
    train_img_folder = os.path.join(output_folder, "train", video_name, "images")
    val_img_folder = os.path.join(output_folder, "val", video_name,"images")
    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(val_img_folder, exist_ok=True)

    # also create labels folders
    train_labels_folder = os.path.join(output_folder, "train", video_name, "labels")
    val_labels_folder = os.path.join(output_folder, "val", video_name, "labels")
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval to sample frames
    interval = total_frames // num_frames

    # Determine the number of frames for training and validation
    num_train_frames = int(num_frames * 0.8)
    num_val_frames = num_frames - num_train_frames

    # Counters for frames saved
    saved_frames = 0
    saved_train_frames = 0
    saved_val_frames = 0

    for frame_count in range(total_frames):
        success, frame = cap.read()
        if not success:
            break  # No more frames to read

        # Save a frame if it's on the interval
        if frame_count % interval == 0:
            if saved_train_frames < num_train_frames:
                frame_filename = os.path.join(train_img_folder, f"frame_{saved_train_frames:05}.jpg")
                saved_train_frames += 1
            else:
                frame_filename = os.path.join(val_img_folder, f"frame_{saved_val_frames:05}.jpg")
                saved_val_frames += 1

            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
            if saved_frames >= num_frames:
                break  # Stop if we've saved the desired number of frames

    cap.release()

# Usage example
video_path = 'C:/Users/msari/Research/Project/snake/dataset/pim/vid012.mp4'
output_folder = 'C:/Users/msari/Research/Project/snake/dataset/pim/op_frames'
num_frames = 100  # Number of frames you want to extract
extract_frames(video_path, output_folder, num_frames)