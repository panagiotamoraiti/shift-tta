import json
import os


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def merge_json_data(json_data, *additional_json_data):
    merged_annotations = []
    # Add original annotations
    merged_annotations.extend(json_data['annotations'])
    # Add additional annotations
    for data in additional_json_data:
        merged_annotations.extend(data['annotations'])

    # Create the updated JSON data
    updated_json_data = {
        'images': json_data.get('images', []) + [img for d in additional_json_data for img in d['images']],
        'categories': json_data.get('categories', []),
        'annotations': merged_annotations,
    }

    return updated_json_data


# json_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
json_path = "/home/panagiota/work/tta/shift-detection-tta/data/cityscapes_foggy/leftImg8bit/train/"
json_data = read_json_file(json_path + 'det_2d_cocoformat_0.02_new.json')
json_data_01 = read_json_file(json_path + 'det_2d_cocoformat_0.01_new.json')
json_data_02 = read_json_file(json_path + 'det_2d_cocoformat_0.02_new.json')
json_data_005 = read_json_file(json_path + 'det_2d_cocoformat_0.005_new.json')

# Merge the JSON data
merged_data = merge_json_data(json_data_005, json_data_01, json_data_02)

file_path = os.path.join(json_path, 'det_2d_cocoformat_all.json')
# Write the modified data back to the JSON file
write_json_file(merged_data, file_path)



'''# Iterate through each item
for item in json_data['images']:
    id_value = item['id']
    new_id = id_value + 500*2
    item['id'] = new_id

# Iterate through each item
for item in json_data['annotations']:
    img_id_value = item['image_id']
    new_img_id_value = img_id_value + 500*2
    item['image_id'] = new_img_id_value

    id_value = item['id']
    new_id_value = id_value + 9792*2
    item['id'] = new_id_value


file_path = os.path.join(json_path, 'det_2d_cocoformat_0.02_new.json')
# Write the modified data back to the JSON file
write_json_file(json_data, file_path)'''



