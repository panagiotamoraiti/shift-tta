#!/usr/bin/env bash

# Define common paths as variables
DATA_DIR="/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/test/front/img_decompressed"
LABELS_FILE="/home/panagiota/work/tta/shift-detection-tta/results/visualize_videos/test_predicted_labels/results.bbox.scalabel.json"
OUTPUT_DIR="/home/panagiota/work/tta/shift-detection-tta/results/visualize_videos/test_predicted_videos"

eval "$(conda shell.bash hook)"
conda activate shift_dev

cd /home/panagiota/work/tta/shift-dev

# Use variables in the commands
python -m shift_dev.vis.video 4ae6-29cf -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 4e8b-2832 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 05a0-5152 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 5d1c-8bae -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 6caf-8e8c -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 7a73-541d -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 9c23-b016 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 64ec-1924 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 270c-40b7 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 0474-0263 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 659d-e134 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 2300-6335 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 2307-fd36 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 4944-d316 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 9825-582b -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video a821-d537 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video b968-e9c9 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video e3a7-8f48 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video eb55-22a5 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video efe9-7187 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front

