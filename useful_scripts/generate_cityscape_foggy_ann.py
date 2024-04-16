import json
import os


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# json_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
json_path = "/home/panagiota/work/tta/shift-detection-tta/data/cityscapes_foggy/leftImg8bit/train/"
json_data = read_json_file(json_path + 'det_2d_cocoformat_0.01.json')


# Keep track of changes
changed_files = []

# Iterate through each item
for item in json_data['images']:
    file_name = item['file_name']
    # Split the file name at the last underscore
    parts = file_name.rsplit('_', 1)
    if len(parts) == 2:
        prefix, suffix = parts
        # Check if the suffix contains '0.02' and replace it with '0.05'
        if '0.02' in suffix:
            new_suffix = suffix.replace('0.02', '0.01')
            new_file_name = prefix + '_' + new_suffix
            item['file_name'] = new_file_name
            changed_files.append(new_file_name)

file_path = os.path.join(json_path, 'det_2d_cocoformat_0.01.json')
# Write the modified data back to the JSON file
write_json_file(json_data, file_path)

print("Changes written to the JSON file:")
for changed_file in changed_files:
    print(changed_file)
