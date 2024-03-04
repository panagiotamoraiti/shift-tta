import os
import json
from PIL import Image, ImageDraw, ImageFont


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def visualize_bboxes_save_images(json_data, source_path, output_path):
    for frame in json_data['frames']:
        # Load image
        image_name = frame['name']
        video_name = frame['videoName']
        image_path = os.path.join(source_path + "img_decompressed", video_name, image_name)
        image = Image.open(image_path)

        # Draw bounding boxes and labels
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 20)

        for label in frame['labels']:
            x1, y1 = label['box2d']['x1'], label['box2d']['y1']
            x2, y2 = label['box2d']['x2'], label['box2d']['y2']
            label_text = label['category']
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)  # Adjust width of bounding box
            draw.rectangle((x1, y1, x1, y1), fill="red")  # Background for text
            draw.text((x1, y1), label_text, fill="white", font=font, width=2)  # Adjust text color

        # Save modified image
        # Output folder for saving images
        output_folder = os.path.join(output_path)
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        output_image_path = os.path.join(output_folder, video_name)
        os.makedirs(output_image_path, exist_ok=True)
        output_image_path = os.path.join(output_image_path, image_name)
        image.save(output_image_path)


val_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
val_output_path = '/home/panagiota/work/tta/processed_data/visualize_bboxes/val/images'
json_data_val = read_json_file(val_path + "det_2d.json")
visualize_bboxes_save_images(json_data_val, val_path, val_output_path)
