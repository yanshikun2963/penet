#!/bin/bash
# CAPE-SGG Variant: varA_cb_loss
MODE=${1:-predcls}
GPU=${2:-0}
export PYTHONPATH=$(pwd):$PYTHONPATH

CONFIG_FILE="configs/cape_sgg_predcls.yaml"
GLOVE_DIR="./datasets/vg/"
CLIP_EMBED_PATH="./datasets/vg/clip_embeddings.pt"

if [ ! -f "$CLIP_EMBED_PATH" ]; then
    echo "=== Precomputing CLIP embeddings ==="
    python tools/clip_precompute.py --output_dir ./datasets/vg/
    if [ $? -ne 0 ]; then
        python tools/clip_precompute.py --use_openai_clip --output_dir ./datasets/vg/
    fi
fi

echo "=== varA_cb_loss: Mode=$MODE, GPU=$GPU ==="

if [ "$MODE" == "predcls" ]; then
    CUDA_VISIBLE_DEVICES=$GPU torchrun \
        --master_port 10025 --nproc_per_node=1 \
        tools/relation_train_net.py \
        --config-file "$CONFIG_FILE" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR "CAPEPrototypeEmbeddingNetwork" \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
        SOLVER.IMS_PER_BATCH 8 \
        TEST.IMS_PER_BATCH 2 \
        SOLVER.BASE_LR 0.001 \
        SOLVER.WARMUP_FACTOR 0.01 \
        SOLVER.WARMUP_ITERS 1000 \
        SOLVER.MAX_ITER 60000 \
        SOLVER.SCHEDULE.TYPE "WarmupMultiStepLR" \
        SOLVER.STEPS "(28000, 48000)" \
        SOLVER.VAL_PERIOD 2000 \
        SOLVER.CHECKPOINT_PERIOD 2000 \
        SOLVER.PRE_VAL False \
        GLOVE_DIR "$GLOVE_DIR" \
        OUTPUT_DIR "./output/varA_cb_loss"
else
    echo "Only predcls mode for now"; exit 1
fi
echo "=== Done ==="
