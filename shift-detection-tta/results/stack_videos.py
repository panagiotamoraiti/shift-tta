import os
import cv2

def add_titles(frame1, frame2):
    # Add titles to the frames with white background and black letters
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.5
    title_thickness = 5
    title_bg_color = (0, 0, 0)  
    title_color = (255, 255, 255)  

    # Get the size of the text for each video
    title1_size = cv2.getTextSize("Ground Truth", title_font, title_scale, title_thickness)[0]
    title2_size = cv2.getTextSize("Mean Teacher", title_font, title_scale, title_thickness)[0]

    # Calculate positions for titles
    title1_x = (frame1.shape[1] - title1_size[0]) // 2
    title1_y = 60
    title2_x = (frame2.shape[1] - title2_size[0]) // 2
    title2_y = 60

    # Create white background for titles
    cv2.rectangle(frame1, (0, 0), (frame1.shape[1], title1_size[1] + 40), title_bg_color, -1)
    cv2.rectangle(frame2, (0, 0), (frame2.shape[1], title2_size[1] + 40), title_bg_color, -1)

    # Put text on the frames
    cv2.putText(frame1, "Ground Truth", (title1_x, title1_y), title_font, title_scale, title_color, title_thickness)
    cv2.putText(frame2, "Mean Teacher", (title2_x, title2_y), title_font, title_scale, title_color, title_thickness)

def stack_videos(video1_path, video2_path, output_path):
    # Open the videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    output_name = os.path.basename(output_path)
    print(f"Saving video: {output_name}")

    # Get video properties
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Add titles to the frames
        add_titles(frame1, frame2)

        # Stack frames horizontally
        combined_frame = cv2.hconcat([frame1, frame2])

        # Write the combined frame
        out.write(combined_frame)

    # Release everything when done
    cap1.release()
    cap2.release()
    out.release()
    

def process_folders(folder1_path, folder2_path, output_folder):
    video1_files = os.listdir(folder1_path)
    video2_files = os.listdir(folder2_path)

    for video1_file, video2_file in zip(video1_files, video2_files):
        video1_path = os.path.join(folder1_path, video1_file)
        video2_path = os.path.join(folder2_path, video2_file)
        output_path = os.path.join(output_folder, video1_file)  

        stack_videos(video1_path, video2_path, output_path)

if __name__ == "__main__":
    folder1_path = "/home/panagiota/work/tta/shift-detection-tta/results/visualize_videos/val_ground_truth_videos"
    folder2_path = "/home/panagiota/work/tta/shift-detection-tta/results/visualize_videos/val_predicted_videos"
    output_folder = "/home/panagiota/work/tta/shift-detection-tta/results/visualize_videos/val_videos_gt_and_pred"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    process_folders(folder1_path, folder2_path, output_folder)

