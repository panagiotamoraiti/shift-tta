CONFIG=configs/continuous/dino/no_adap_yolox_dino/yolox_x_8xb4-24e_coco_no_adaptation_dino.py
CKPT=https://dl.cv.ethz.ch/shift/challenge2023/test_time_adaptation/checkpoints/yolox_x_8xb4-24e_shift_clear_daytime.pth
WORK_DIR=work_dirs/continuous/dino/no_adap_yolox_dino/val/yolox_x_8xb4-24e_coco

declare -a CFG_OPTIONS=(
     "test_evaluator.0.outfile_prefix=${WORK_DIR}/results"
)

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=true
python tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT} \
     --work-dir ${WORK_DIR} \
     --cfg-options ${CFG_OPTIONS[@]}
