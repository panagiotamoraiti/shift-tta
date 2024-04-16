import json

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# Define the mapping of old category IDs to new category IDs
# old_id -> new_id
# class_names = {1: "person", 2: "rider", 3: "car", 4: "train", 5: "motorcycle", 6: "bicycle", 7: "truck", 8: "bus"} to keep that
# class_names = {1: "person", 2: "car", 3: "train", 4: "rider", 5: "truck", 6: "motorcycle", 7: "bicycle", 8: "bus"}
id_mapping = {1: 1, 2: 3, 3: 4, 4: 2, 5: 7, 6: 5, 7: 6, 8: 8}

# Load the JSON data
json_path = "/home/panagiota/work/tta/shift-detection-tta/data/cityscapes_foggy/leftImg8bit/train/"
json_file = 'det_2d_cocoformat_all_new.json'
json_data = read_json_file(json_path + json_file)

# Iterate through annotations and update category_id
for annotation in json_data['annotations']:
    old_category_id = annotation['category_id']
    if old_category_id in id_mapping:
        new_category_id = id_mapping[old_category_id]
        annotation['category_id'] = new_category_id

# Write the modified data back to the JSON file
write_json_file(json_path + json_file, json_data)

json_data = read_json_file(json_path + json_file)

# Define the mapping of old category names to new category names
class_names = ["person", "rider", "car", "train", "motorcycle", "bicycle", "truck", "bus"]

# Update the categories list
i = 0
for category in json_data['categories']:
    category['name'] = class_names[i]
    i += 1

# Write the modified data back to the JSON file
write_json_file(json_path + json_file, json_data)
