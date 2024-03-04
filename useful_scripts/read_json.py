import json


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def print_video_attributes(json_data, clear_daytime=False):
    video_names_printed = set()
    for frame in json_data['frames']:
        video_name = frame['videoName']
        shift_type = frame['attributes']['shift_type']
        shift_length = frame['attributes']['shift_length']

        if video_name not in video_names_printed:
            weather_coarse = frame['attributes']['weather_coarse']
            timeofday_coarse = frame['attributes']['timeofday_coarse']

            if not clear_daytime:
                print(f"{len(video_names_printed) + 1}. Video Name: {video_name}, Weather Coarse: {weather_coarse}, "
                      f"Time of Day Coarse: {timeofday_coarse}, Shift type: {shift_type}, Shift Length: {shift_length}")
            else:
                if weather_coarse == "clear" and timeofday_coarse == "daytime":
                    print(
                        f"{len(video_names_printed) + 1}. Video Name: {video_name}, Weather Coarse: {weather_coarse}, "
                        f"Time of Day Coarse: {timeofday_coarse}, Shift type: {shift_type}, Shift Length: {shift_length}")

            video_names_printed.add(video_name)


def print_image_attributes(json_data):
    video_number = 0
    prev_video_name = None
    for frame in json_data['frames']:
        video_name = frame['videoName']
        image_name = frame['name']
        weather_coarse = frame['attributes']['weather_coarse']
        timeofday_coarse = frame['attributes']['timeofday_coarse']
        shift_type = frame['attributes']['shift_type']
        shift_length = frame['attributes']['shift_length']

        if prev_video_name != video_name:
            prev_video_name = video_name
            video_number += 1

        # if video_number == 1:
        print(f"{video_number}. Video Name: {video_name}, Image Name: {image_name}, Weather Coarse: {weather_coarse}, "
              f"Time of Day Coarse: {timeofday_coarse}, Shift type: {shift_type}, Shift Length: {shift_length}")


def print_unique_attributes(json_data):
    weather_coarse_set = set()
    time_of_day_coarse_set = set()
    weather_fine_set = set()
    time_of_day_fine_set = set()
    shift_type_set = set()
    view_set = set()
    town_set = set()

    for frame in json_data['frames']:
        attributes = frame['attributes']
        weather_coarse_set.add(attributes['weather_coarse'])
        time_of_day_coarse_set.add(attributes['timeofday_coarse'])
        weather_fine_set.add(attributes['weather_fine'])
        time_of_day_fine_set.add(attributes['timeofday_fine'])
        shift_type_set.add(attributes.get('shift_type', ''))  # This handles the case where 'shift_type' might not exist

        view_set.add(attributes['view'])
        town_set.add(attributes['town'])

    print("Unique values for Weather Coarse:")
    for weather_coarse in weather_coarse_set:
        print(weather_coarse)

    print("\nUnique values for Time of Day Coarse:")
    for time_of_day_coarse in time_of_day_coarse_set:
        print(time_of_day_coarse)

    print("\nUnique values for Weather Fine:")
    for weather_fine in weather_fine_set:
        print(weather_fine)

    print("\nUnique values for Time of Day Fine:")
    for time_of_day_fine in time_of_day_fine_set:
        print(time_of_day_fine)

    print("\nUnique values for Shift type:")
    for shift_type in shift_type_set:
        print(shift_type)

    print("\nUnique values for view:")
    for view in view_set:
        print(view)

    print("\nUnique values for town:")
    for town in town_set:
        print(town)


json_val_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/"
json_test_path = "/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/test/front/"

json_data_val = read_json_file(json_val_path + 'det_2d.json')
json_data_test = read_json_file(json_test_path + 'det_2d.json')

print_video_attributes(json_data_val, clear_daytime=True)
# print_video_attributes(json_data_val)
# print_image_attributes(json_data_val)
# print_unique_attributes(json_data_val)

# print_video_attributes(json_data_test, clear_daytime=True)
# print_video_attributes(json_data_test)
# print_image_attributes(json_data_test)
# print_unique_attributes(json_data_test)
