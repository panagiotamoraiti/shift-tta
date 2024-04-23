import os
import json

# Path to the directory containing the COCO JSON file
coco_json_file = "merged_annotations.json"

# Path to the file containing image names for training
train_txt_file = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/train.txt"
test_txt_file = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/test.txt"

# Output directory for the new sorted JSON file
output_directory = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/"

# Read the list of image names from the train.txt file
with open(train_txt_file, "r") as file:
    train_image_names = file.read().splitlines()
    
# Read the list of image names from the test.txt file
with open(test_txt_file, "r") as file:
    test_image_names = file.read().splitlines()

# Load the COCO JSON file
with open(coco_json_file, "r") as file:
    coco_dataset = json.load(file)

# Filter out only the images listed in the train.txt file
train_images = [image for image in coco_dataset["images"] if image["file_name"] in train_image_names]
test_images = [image for image in coco_dataset["images"] if image["file_name"] in test_image_names]

# Filter out only the annotations corresponding to the train images
train_annotations = [annotation for annotation in coco_dataset["annotations"] if annotation["image_id"] in [image["id"] for image in train_images]]
test_annotations = [annotation for annotation in coco_dataset["annotations"] if annotation["image_id"] in [image["id"] for image in test_images]]

# Sort the images and annotations lists based on the order of image names in the text files
train_images = sorted(train_images, key=lambda x: train_image_names.index(x["file_name"]))
train_annotations = sorted(train_annotations, key=lambda x: train_image_names.index([image["file_name"] for image in train_images if image["id"] == x["image_id"]][0]))
test_images = sorted(test_images, key=lambda x: test_image_names.index(x["file_name"]))
test_annotations = sorted(test_annotations, key=lambda x: test_image_names.index([image["file_name"] for image in test_images if image["id"] == x["image_id"]][0]))

# Create a new COCO dataset dictionary containing only the train images and annotations
train_coco_dataset = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": coco_dataset["categories"]
}

test_coco_dataset = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": coco_dataset["categories"]
}

# Save the new sorted JSON file containing only the train images and annotations
output_file_path_train = os.path.join(output_directory, "trainnew.json")
with open(output_file_path_train, "w") as file:
    json.dump(train_coco_dataset, file, indent=4)
    
output_file_path_test = os.path.join(output_directory, "rain.json")
with open(output_file_path_test, "w") as file:
    json.dump(test_coco_dataset, file, indent=4)

print(f"Train JSON file saved to: {output_file_path_train}")
print(f"Test JSON file saved to: {output_file_path_test}")

