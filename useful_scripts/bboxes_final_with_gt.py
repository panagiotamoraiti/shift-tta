import os
import json
import cv2
from PIL import Image, ImageDraw, ImageFont

fps = 10


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def visualize_bboxes_save_images(json_data, source_path, output_path, colour):
    for frame in json_data['frames']:
        # Load image
        image_name = frame['name']
        video_name = frame['videoName']
        image_path = os.path.join(source_path, video_name, image_name)
        image = Image.open(image_path)

        # Draw bounding boxes and labels
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 20)

        for label in frame['labels']:
            x1, y1 = label['box2d']['x1'], label['box2d']['y1']
            x2, y2 = label['box2d']['x2'], label['box2d']['y2']
            label_text = label['category']
            if x2 >= x1:
                draw.rectangle((x1, y1, x2, y2), outline=colour, width=2)  # Adjust width of bounding box
                draw.rectangle((x1, y1, x1, y1), fill=colour)  # Background for text
            else:
                x1, x2 = x2, x1
                draw.rectangle((x1, y1, x2, y2), outline=colour, width=2)  # Adjust width of bounding box
                draw.rectangle((x1, y1, x1, y1), fill=colour)  # Background for text
            draw.text((x1, y1), label_text, fill=colour, font=font, width=2)  # Adjust text color

        # Save modified image
        # Output folder for saving images
        output_folder = os.path.join(output_path)
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        output_image_path = os.path.join(output_folder, video_name)
        os.makedirs(output_image_path, exist_ok=True)
        output_image_path = os.path.join(output_image_path, image_name)
        image.save(output_image_path)


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

# Path to images and videos
path = "/home/panagiota/work/tta/processed_data/visualize_bboxes/val_ground_truth/"

# Save images with mean teacher
val_output_path = '/home/panagiota/work/tta/processed_data/visualize_bboxes/mean_teacher/val_mean_teacher/images_with_gt'
json_path = '/home/panagiota/work/tta/processed_data/visualize_bboxes/mean_teacher/val_mean_teacher/'
json_data_val = read_json_file(json_path + "results'].bbox.scalabel.json")
visualize_bboxes_save_images(json_data_val, path, val_output_path, colour="cyan")

# Save video
img_path = '/home/panagiota/work/tta/processed_data/visualize_bboxes/mean_teacher/val_mean_teacher/images_with_gt'
val_output_path = '/home/panagiota/work/tta/processed_data/visualize_bboxes/mean_teacher/val_mean_teacher/videos_with_gt'
images_to_video(json_data_val, img_path, val_output_path)

