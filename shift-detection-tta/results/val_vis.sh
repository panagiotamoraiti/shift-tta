#!/usr/bin/env bash

# Define common paths as variables
DATA_DIR="/home/panagiota/work/tta/shift-detection-tta/data/shift/continuous/videos/1x/val/front/img_decompressed"
LABELS_FILE="/home/panagiota/work/tta/shift-detection-tta/results/visualize_videos/val_predicted_labels/results.bbox.scalabel.json"
OUTPUT_DIR="/home/panagiota/work/tta/shift-detection-tta/results/visualize_videos/val_predicted_videos"

eval "$(conda shell.bash hook)"
conda activate shift_dev

cd /home/panagiota/work/tta/shift-dev

# Use variables in the commands
python -m shift_dev.vis.video 28c6-f5ec -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 620f-1d49 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video 6228-8e74 -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video c9b1-1b8e -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video e32b-792e -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front
python -m shift_dev.vis.video e068-112f -d "$DATA_DIR" -l "$LABELS_FILE" -o "$OUTPUT_DIR" --view front

