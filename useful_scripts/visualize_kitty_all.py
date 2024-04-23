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
        #draw.rectangle((x1, y1, x1+25, y1+45), fill="black")  # Background for text
        draw.text((x1, y1), category_name, fill="white", font=font, width=2)  # Adjust text color
        # draw.text((x1, y1+20), str(round(score,2)), fill="white", font=font, width=2)  # Adjust text color

    # Save modified image
    image.save(output_path)

# CITYSCAPES
# json_path = "/home/panagiota/work/tta/shift-detection-tta/data/cityscapes_foggy/leftImg8bit/val/"
# json_data = read_json_file(json_path + 'det_2d_cocoformat_0.02_new.json')
# img_path = "/home/panagiota/work/tta/shift-detection-tta/data/cityscapes_foggy/leftImg8bit/val/"
#
# path = '/home/panagiota/work/tta/processed_data/visualize_bboxes/cityscapes/pred_0.02/'
# json_data1 = read_json_file(path + 'results.bbox.json')
#
# class_names = {1: "person", 2: "rider", 3: "car", 4: "train", 5: "motorcycle", 6: "bicycle", 7: "truck", 8: "bus"}


# KITTI
path = "/home/panagiota/work/tta/processed_data/visualize_bboxes/kitti/fog-rain-snow-clear/"
img_path = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training"

json_path = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/"
json_data = read_json_file(json_path + 'fog-rain-snow-clear_new.json')

class_names = {1: "Car", 2: "Van", 3: "Pedestrian", 4: "Cyclist", 5: "Truck", 6: "Misc", 7: "Tram",
               8: "Person_sitting", 9: "DontCare"}


# Iterate through each item
for item in json_data['images']:
    file_name = item['file_name']
    parts = file_name.split('/')
    image_name = parts[-1]
    img_id = item['id']

    if "fog" in file_name:
        weather = "fog"
    if "rain" in file_name:
        weather = "rain"
    if "snow" in file_name:
        weather = "snow"
    if "image_2" in file_name:
        weather = "clear"

    if weather == "fog" or weather == "rain" or weather == "snow":
        continue
    output_path = os.path.join(path, weather)
    output_path = os.path.join(output_path, image_name)

    # For every image draw bboxes and labels
    bboxes = []
    labels = []
    scores = []

    for el in json_data['annotations']: ### for el in json_data:
        if el['image_id'] == img_id:
            # print(el)
            bboxes.append(el["bbox"])
            labels.append(class_names[el["category_id"]])
            #scores.append(el["score"])
    # print(image_name)
    # print(labels)
    # print(bboxes)


    # Visualize bounding boxes and categories
    # visualize_bboxes_save_images(file_name, bboxes, labels, scores, output_path)

    file_name_new = img_path + file_name
    print(file_name)

    visualize_bboxes_save_images(file_name_new, bboxes, labels, output_path)
