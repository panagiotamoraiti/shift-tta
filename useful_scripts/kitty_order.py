import json
import os


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


json_path = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/"
json_file = 'fog-rain-snow-clear_final.json'

json_data = read_json_file(json_path + json_file)

# for i, item in enumerate(json_data['images'], start=1):
#     item['id'] = i

for i, item in enumerate(json_data['annotations'], start=1):
    item['id'] = i
    print(i)

# # Initialize a dictionary to store counts for each unique image_id
# image_id_counts = []
#
# id = 0
# # Iterate through the JSON data and modify image_id
# for item in json_data["annotations"]:
#     image_id = item['image_id']
#     if image_id not in image_id_counts:
#         id += 1
#     item['image_id'] = id
#     image_id_counts.append(image_id)

# for item in json_data["annotations"]:
#     print(item)


file_path = os.path.join(json_path, 'fog-rain-snow-clear_final.json')
write_json_file(json_data, file_path)

