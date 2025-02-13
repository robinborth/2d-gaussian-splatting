#!/bin/bash

SCAN=105
ROOT=/home/borth/2d-gaussian-splatting
OUTPUT=$ROOT/output/
DATA_DIR=$ROOT/data/DTU/scan$SCAN
MODEL_DIR=/home/borth/2d-gaussian-splatting/logs/2025-02-13/11-10-45
EVAL_INPUT_MESH=$MODEL_DIR/train/ours_10/fuse_post.ply
EVAL_OUTPUT_DIR=$MODEL_DIR/train/ours_10
EVAL_OUTPUT_MESH=$EVAL_OUTPUT_DIR/culled_mesh.ply
MASK_DIR=$ROOT/data/DTU
DTU_DIR=$ROOT/data/Offical_DTU_Dataset

# Train
# python train.py \
#      -s $DATA_DIR \
#      -m $MODEL_DIR \
#      -r 2 \
#      --depth_ratio 1 \
#      --lambda_normal 0.05 \
#      --lambda_dist 1000

# Render
# python render.py \
#      -s $DATA_DIR \
#      -m $MODEL_DIR \
#      -r 2 \
#      --skip_test \
#      --depth_ratio 1 \
#      --mesh_res 1024 

# Evaluate
# python scripts/eval_dtu/evaluate_single_scene.py \
#      --input_mesh $EVAL_INPUT_MESH \
#      --scan_id $SCAN \
#      --output_dir $EVAL_OUTPUT_DIR \
#      --mask_dir $MASK_DIR \
#      --DTU $DTU_DIR


python scripts/eval_dtu/eval_old.py \
     --data $EVAL_OUTPUT_MESH \
     --scan $SCAN \
     --mode mesh \
     --dataset_dir $DTU_DIR \
     --vis_out_dir $EVAL_OUTPUT_DIR

