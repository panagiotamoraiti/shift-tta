# CONFIG=configs/source/yolox/yolox_x_8xb4-24e_kitti.py
CONFIG=configs/source/yolox/amp_yolox_x_8xb4-24e_kitti.py

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/train.py \
python tools/train.py \
     ${CONFIG}
