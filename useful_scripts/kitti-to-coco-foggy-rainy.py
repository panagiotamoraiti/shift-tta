import os
import json
from PIL import Image

# Path to the folder containing KITTI format text files
kitti_folder = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/data_object_label_2/training/label_2"

# Initialize COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "Car"},
        {"id": 2, "name": "Van"},
        {"id": 3, "name": "Pedestrian"},
        {"id": 4, "name": "Cyclist"},
        {"id": 5, "name": "Truck"},
        {"id": 6, "name": "Misc"},
        {"id": 7, "name": "Tram"},
        {"id": 8, "name": "Person_sitting"},
        {"id": 9, "name": "DontCare"}
        # Add more categories if necessary
    ]
}

# Function to parse KITTI format annotations and convert to COCO format
def parse_kitti_annotations(file_path, original_image_size, resized_image_size):
    annotations = []
    original_width, original_height = original_image_size
    resized_width, resized_height = resized_image_size
    width_ratio = resized_width / original_width
    height_ratio = resized_height / original_height
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split(' ')
            category = data[0]
            x1, y1, x2, y2 = [float(val) for val in data[4:8]]  # Extract bounding box coordinates
            
            # Adjust bounding box coordinates based on the resizing ratio
            x1_resized = x1 * width_ratio
            y1_resized = y1 * height_ratio
            x2_resized = x2 * width_ratio
            y2_resized = y2 * height_ratio
            
            x_resized, y_resized = min(x1_resized, x2_resized), min(y1_resized, y2_resized)
            w_resized, h_resized = abs(x2_resized - x1_resized), abs(y2_resized - y1_resized)
            
            annotations.append({
                "category_id": 9 if category == "DontCare" else next(item["id"] for item in coco_dataset["categories"] if item["name"] == category),
                "bbox": [x_resized, y_resized, w_resized, h_resized],
                "area": w_resized * h_resized,
                "iscrowd": 0,
                "ignore": 0
            })
    return annotations
    
image_folder = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/image_2"

# Iterate over each file in the KITTI folder
for filename in sorted(os.listdir(kitti_folder)):
    if filename.endswith(".txt"):
        image_id = filename.split('.')[0]
        image_path = os.path.join(image_folder, f"{image_id}.png")
        image = Image.open(image_path)
        image_width, image_height = image.size
        w, h = 1216, 352
        annotations = parse_kitti_annotations(os.path.join(kitti_folder, filename), (image_width, image_height), (w, h))  # Corrected the arguments
        # Add image information to COCO dataset
        coco_dataset["images"].append({
            "id": int(image_id),
            "file_name": f"{image_id}.png",  # Assuming images are named with their ID and have JPG extension
            "height": image_height,
            "width": image_width
        })
        # Add annotations to COCO dataset
        for annotation in annotations:
            annotation["image_id"] = int(image_id)
            annotation["id"] = len(coco_dataset["annotations"]) + 1
            coco_dataset["annotations"].append(annotation)

# Sort the annotations by image_id and id
coco_dataset["annotations"] = sorted(coco_dataset["annotations"], key=lambda x: (x["image_id"], x["id"]))

# Save the merged COCO annotations to a JSON file
with open("merged_annotations.json", "w") as json_file:
    json.dump(coco_dataset, json_file, indent=4)

