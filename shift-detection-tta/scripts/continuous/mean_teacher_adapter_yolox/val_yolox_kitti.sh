CONFIG=configs/continuous/mean_teacher_adapter_yolox/yolox_x_8xb4-24e_kitti.py
CKPT=/home/panagiota/work/tta/shift-detection-tta/checkpoints/yolox_x_8xb4-24e_kitti.pth
WORK_DIR=work_dirs/continuous/mean_teacher_yolox/val/yolox_x_8xb4-24e_kitti

declare -a CFG_OPTIONS=(
     "test_evaluator.0.outfile_prefix=${WORK_DIR}/results"
)

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT} \
     --work-dir ${WORK_DIR} \
     --cfg-options ${CFG_OPTIONS[@]}
