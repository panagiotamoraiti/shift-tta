#!/usr/bin/env bash

./scripts/continuous/mean_teacher_adapter_yolox/val_yolox_cityscapes.sh
./scripts/continuous/mean_teacher_adapter_yolox/val_yolox_shift_from_clear_daytime.sh 
./scripts/continuous/mean_teacher_adapter_yolox/val_yolox_kitti_baseline.sh 

shutdown -h +1
