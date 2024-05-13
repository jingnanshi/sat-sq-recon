SPLIT=validation
SAVE_DIR=figures
NUM_PRIM=8
PRETRAIN=output/model_m8_rgb_v1_1.pth.tar

python tools/evaluate.py --cfg experiments/config_docker.yaml --split ${SPLIT} --save_dir ${SAVE_DIR} \
        MODEL.NUM_MAX_PRIMITIVES ${NUM_PRIM} MODEL.PRETRAIN_FILE ${PRETRAIN}
