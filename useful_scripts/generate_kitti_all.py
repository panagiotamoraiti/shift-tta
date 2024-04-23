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


json_path = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/"


# json_data = read_json_file(json_path + 'test.json')
#
# # Iterate through each item
# for item in json_data['images']:
#     id_value = item['id']
#     new_id = id_value + 7480*3
#     item['id'] = new_id
#
# # Iterate through each item
# for item in json_data['annotations']:
#     img_id_value = item['image_id']
#     new_img_id_value = img_id_value + 7480*3
#     item['image_id'] = new_img_id_value
#
#     id_value = item['id']
#     new_id_value = id_value + 51865*3
#     item['id'] = new_id_value
#
#
# file_path = os.path.join(json_path, 'test_new.json')
# # Write the modified data back to the JSON file
# write_json_file(json_data, file_path)

json_data1 = read_json_file(json_path + 'fog.json')
json_data2 = read_json_file(json_path + 'rain_new.json')
json_data3 = read_json_file(json_path + 'snow_new.json')
json_data = read_json_file(json_path + 'test_new.json')

# Merge the JSON data
merged_data = merge_json_data(json_data1, json_data2, json_data3, json_data)

file_path = os.path.join(json_path, 'fog-rain-snow-clear_final.json')
# Write the modified data back to the JSON file
write_json_file(merged_data, file_path)
