VALSPLIT=validation
TESTSPLIT=test
SAVE_DIR=eval_output
NUM_PRIM=8
PRETRAIN=output/model_m8_rgb_v1_1.pth.tar

python tools/evaluate_dataset.py --cfg experiments/config_docker.yaml --split ${VALSPLIT} --save_dir "${SAVE_DIR}/validation" \
        MODEL.NUM_MAX_PRIMITIVES ${NUM_PRIM} MODEL.PRETRAIN_FILE ${PRETRAIN}
python tools/evaluate_dataset.py --cfg experiments/config_docker.yaml --split ${TESTSPLIT} --save_dir "${SAVE_DIR}/test" \
        MODEL.NUM_MAX_PRIMITIVES ${NUM_PRIM} MODEL.PRETRAIN_FILE ${PRETRAIN}
