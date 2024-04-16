import os
import json
from PIL import Image, ImageDraw, ImageFont

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def visualize_bboxes_save_images(image_path, bboxes, labels, output_path):
    image = Image.open(image_path)

    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)

    for bbox, category_name in zip(bboxes, labels):
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)  # Adjust width of bounding box
        draw.rectangle((x1, y1, x1, y1), fill="red")  # Background for text
        draw.text((x1, y1), category_name, fill="white", font=font, width=2)  # Adjust text color

    # Save modified image
    image.save(output_path)


# json_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
json_path = "/home/panagiota/work/tta/shift-detection-tta/data/cityscapes_foggy/leftImg8bit/val/"
json_data = read_json_file(json_path + 'det_2d_cocoformat_0.005_new.json')
path = '/home/panagiota/work/tta/processed_data/visualize_bboxes/cityscapes/foggy_val_0.005'

class_names = {1: "person", 2: "rider", 3: "car", 4: "train", 5: "motorcycle", 6: "bicycle", 7: "truck", 8: "bus"}
# class_names = {1: "person", 2: "car", 3: "train", 4: "rider", 5: "truck", 6: "motorcycle", 7: "bicycle", 8: "bus"}


# Iterate through each item
for item in json_data['images']:
    file_name = item['file_name']
    parts = file_name.split('/')
    image_name = parts[-1]
    img_id = item['id']
    output_path = os.path.join(path, image_name)
    # print(image_name)

    # For every image draw bboxes and labels
    bboxes = []
    labels = []
    for el in json_data['annotations']:
        if el['image_id'] == img_id:
            # print(el)
            bboxes.append(el["bbox"])
            labels.append(class_names[el["category_id"]])
    # print(image_name)
    # print(labels)

    file_name = os.path.join(json_path, file_name)

    # Visualize bounding boxes and categories
    visualize_bboxes_save_images(file_name, bboxes, labels, output_path)
