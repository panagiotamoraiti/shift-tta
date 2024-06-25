CONFIG=configs/continuous/prototypes/no_adap_yolox/yolox_x_8xb4-24e_cityscapes_prototypes.py
CKPT=/home/panagiota/work/tta/shift-detection-tta/checkpoints/yolox_x_8xb4-24e_cityscapes.pth
WORK_DIR=work_dirs/continuous/prototypes/no_adap_yolox_prot/val/yolox_x_8xb4-24e_cityscapes

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
