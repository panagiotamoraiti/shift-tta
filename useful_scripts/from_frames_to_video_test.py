import os
import json
import cv2

fps = 10


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def images_to_video(json_data, source_path, output_path):
    # Dictionary to keep track of initialized video writers
    video_writers = {}

    # Iterate through frames
    for frame in json_data['frames']:
        # Load image
        image_name = frame['name']
        video_name = frame['videoName']
        image_path = os.path.join(source_path, video_name, image_name)

        # Read the first image to get its size
        image = cv2.imread(image_path)
        frame_height, frame_width, _ = image.shape

        # Check if video writer is already initialized for this video name
        if video_name not in video_writers:
            # Initialize video writer
            output_folder = os.path.join(output_path)
            os.makedirs(output_folder, exist_ok=True)
            output_video_path = os.path.join(output_folder, f"{video_name}.mp4")
            video_writers[video_name] = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Get the video writer for this video name
        video_writer = video_writers[video_name]

        # Write frame to video
        video_writer.write(image)
        print(f"Frame {image_name} added to video {video_name}.")

    # Release all video writers
    for video_writer in video_writers.values():
        video_writer.release()

    print("All videos saved.")


val_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
val_output_path = '~/work/tta/processed_data/visualize_bboxes/no_adaptation_baseline/val_no_adaptation/videos'
images_bboxes_path = "/home/panagiota/work/tta/processed_data/visualize_bboxes/val/images/"
json_data_val = read_json_file(val_path + "det_2d.json")
images_to_video(json_data_val, images_bboxes_path, val_output_path)
