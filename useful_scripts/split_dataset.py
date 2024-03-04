import os
import shutil
import json


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def split_videos_by_attributes(json_data, source_folder, destination_folder):
    processed_videos = set()  # Keep track of processed videos
    for frame in json_data['frames']:
        video_name = frame['videoName']
        weather_coarse = frame['attributes']['weather_coarse']
        timeofday_coarse = frame['attributes']['timeofday_coarse']
        shift_type = frame['attributes']['shift_type']

        # Check if the video has been processed already
        if video_name in processed_videos:
            continue

        # Create destination folder based on attributes
        if timeofday_coarse == "dawn/dusk":
            final_folder = os.path.join(destination_folder, f"{weather_coarse}_dawn_dusk", shift_type)
        else:
            final_folder = os.path.join(destination_folder, f"{weather_coarse}_{timeofday_coarse}", shift_type)
        os.makedirs(final_folder, exist_ok=True)

        # Convert backslashes to forward slashes for consistency
        source_video_file = os.path.join(source_folder, video_name + ".mp4")
        if os.path.exists(source_video_file):
            destination_video_file = os.path.join(final_folder, video_name + ".mp4")
            shutil.copy(source_video_file, destination_video_file)
            print(f"Moved '{video_name}' to '{final_folder}'")

        # Mark the video as processed
        processed_videos.add(video_name)


val_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
test_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/test/front/"

json_data_val = read_json_file(val_path + "det_2d.json")
source_folder = val_path + "img"
destination_folder = "/home/panagiota/work/tta/processed_data/videos_val_split"
split_videos_by_attributes(json_data_val, source_folder, destination_folder)

json_data_test = read_json_file(test_path + "det_2d.json")
source_folder = test_path + "img"
destination_folder = "/home/panagiota/work/tta/processed_data/videos_test_split"
split_videos_by_attributes(json_data_test, source_folder, destination_folder)
