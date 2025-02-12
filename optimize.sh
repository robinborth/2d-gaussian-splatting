#!/bin/bash

SCAN=63
ROOT=/home/borth/2d-gaussian-splatting
OUTPUT=$ROOT/output/
DATA_DIR=$ROOT/data/DTU/scan$SCAN
MODEL_DIR=$OUTPUT/11-02-25/scan$SCAN
EVAL_INPUT_MESH=$MODEL_DIR/train/ours_30000/fuse_post.ply
EVAL_OUTPUT_DIR=$MODEL_DIR/train/ours_30000
MASK_DIR=$ROOT/data/DTU
DTU_DIR=$ROOT/data/Offical_DTU_Dataset

# Train
python train.py \
     -s $DATA_DIR \
     -m $MODEL_DIR \
     -r 2 \
     --depth_ratio 1 \
     --lambda_normal 0.05 \
     --lambda_dist 1000

# Render
python render.py \
     -s $DATA_DIR \
     -m $MODEL_DIR \
     -r 2 \
     --skip_test \
     --depth_ratio 1 \
     --mesh_res 1024 

# Evaluate
python scripts/eval_dtu/evaluate_single_scene.py \
     --input_mesh $EVAL_INPUT_MESH \
     --scan_id $SCAN \
     --output_dir $EVAL_OUTPUT_DIR \
     --mask_dir $MASK_DIR \
     --DTU $DTU_DIR