import json


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# json_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
json_path = "/home/panagiota/work/tta/shift-detection-tta/data/cityscapes_foggy/leftImg8bit/train/"
json_file = 'det_2d_cocoformat_0.02_new.json'
json_data = read_json_file(json_path + json_file)


# Print all keys in the JSON data
print("Keys in JSON data:")
for key in json_data.keys():
    print(key)

# Print contents of each key
for key, value in json_data.items():
    print(f"Key: {key}")
    if isinstance(value, list):
        print("Value is a list:")
        for item in value:
            if 'segmentation' in item:
                del item['segmentation']
            print(item)
    elif isinstance(value, dict):
        print("Value is a dictionary:")
        for k, v in value.items():
            if k == 'segmentation':
                continue
            print(f"    {k}: {v}")
    else:
        print(f"Value: {value}")
    print()  # Empty line for separation