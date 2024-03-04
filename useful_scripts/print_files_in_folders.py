import os


def print_files_in_folder(folder_path):
    if os.path.exists(folder_path):
        total_file_count = 0  # Initialize a counter for total files

        # Walk through the directory tree
        for root, dirs, files in os.walk(folder_path):
            print("\nCurrent directory:", root)
            file_count = 0  # Initialize a counter for files in the current directory

            # Print files in the current directory
            for file in files:
                file_count += 1  # Increment the counter for each file
                print(os.path.join(root, file))

            # Print the number of files in the current directory
            print("Total files in", root, ":", file_count)
            total_file_count += file_count  # Update the total file count

        print("\nTotal files found:", total_file_count)  # Print total file count after traversing the directory
    else:
        print("Folder does not exist.")


# folder_path = '/home/panagiota/work/tta/processed_data/videos_val_split'
folder_path = '/home/panagiota/work/tta/processed_data/videos_test_split'
print_files_in_folder(folder_path)
