import os
import json

# Function to read JSON file
def read_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Function to update file names in the JSON data
def update_filenames_with_path(json_data, img_path):
    for entry in json_data["images"]:
        file_name = entry['file_name']
        file_name_parts = file_name.split('/')
        entry['file_name'] = img_path + '/' + file_name_parts[-1]
    return json_data


# Function to write JSON data back to file
def write_json_file(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

# Paths
img_path = "image_2"
json_path = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/"

# Read JSON file
json_data = read_json_file(os.path.join(json_path, 'test_new.json'))

# Update file names with path
json_data = update_filenames_with_path(json_data, img_path)

# Write updated JSON data back to file
write_json_file(os.path.join(json_path, 'test_new.json'), json_data)

