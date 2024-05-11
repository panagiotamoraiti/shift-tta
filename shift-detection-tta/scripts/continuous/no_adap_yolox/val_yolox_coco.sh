CONFIG=configs/continuous/no_adap_yolox/yolox_x_8xb4-24e_coco.py
CKPT=/home/panagiota/work/tta/shift-detection-tta/checkpoints/yolox_x_8xb4-24e_coco.pth
WORK_DIR=work_dirs/continuous/no_adap_yolox/val/yolox_x_8xb4-24e_coco

declare -a CFG_OPTIONS=(
     "test_evaluator.0.outfile_prefix=${WORK_DIR}/results"
)

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
python tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT} \
     --work-dir ${WORK_DIR} \
     --cfg-options ${CFG_OPTIONS[@]}
