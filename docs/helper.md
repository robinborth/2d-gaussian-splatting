# Installation
conda env create -n 2dgs python=3.8.10
conda env update --file environment.yml 
conda activate 2dgs
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization
pip install notebook jupyter tensorboard torch_tb_profiler wandb lightning

<!-- conda install pytorch3d -c pytorch3d -->
<!-- pip install "git+https://github.com/facebookresearch/pytorch3d.git" -->

# Viser Viewer Repo
https://github.com/hwanhuh/2D-GS-Viser-Viewer

# Install the DTU Dataset
https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9

# Extracting the DTU Dataset
tar -xvzf dtu-001.tar.gz 



# DTU-Dataset Evaluation (Bounded Scene)

1) Optimize the DTU object.


## Arguments 
-s: The source path of the model, which is preprocessed in the DTU dataset case.
-m: The output folder where to save the point_cloud with the checkpoints.
-r: The resolution where 1 is full resolution and 2 is just the half.


# Ablations
--depth_ratio: Ranges between [0,1] where 0 is mean depth map and 1 is median depth map.
--lambda_normal: The normal consistency loss, which is default L_n=0.5
--lambda_dist: The distortion loss, which is for bounded sceenes L_d=1000

## Example Script
```bash
python train.py \
     -s /home/borth/2d-gaussian-splatting/data/DTU/scan105 \
     -m output/03_11-02-25/scan105 \
     -r 2 \
     --depth_ratio 1 \
     --lambda_normal 0.05 \
     --lambda_dist 1000
```


2) Extract the Mesh & Render Result

# For Bounded Scenes use the following Hparams
--depth_ratio 1
--voxel_size 0.004
--depth_trunc 0.02

Else, you can specify the mesh_resolution with:
--mesh_res

## Arguments
-s: The source path of the model, which is preprocessed in the DTU dataset case.
-m: The output folder where to save the point_cloud with the checkpoints.
-r: The resolution where 1 is full resolution and 2 is just the half.
--depth_ratio: Ranges between [0,1] where 0 is mean depth map and 1 is median depth map.
--skip_train: Ensures to skip to render the training images from the Gaussian representation.
--skip_test: Ensures to skip to render the test images from the Gaussian representation.

```bash
python render.py \
     -s /home/borth/2d-gaussian-splatting/data/DTU/scan105 \
     -m /home/borth/2d-gaussian-splatting/output/03_11-02-25/scan105 \
     -r 2 \
     --skip_test \
     --depth_ratio 1 \
     --mesh_res 1024 \
     # --voxel_size 0.004 \
     # --depth_trunc 0.02
```

3) Evaluate the Extracted Mesh

```bash
python scripts/eval_dtu/evaluate_single_scene.py \
     --input_mesh /home/borth/2d-gaussian-splatting/output/03_11-02-25/scan105/train/ours_30000/fuse_post.ply  \
     --scan_id 105 \
     --output_dir /home/borth/2d-gaussian-splatting/output/03_11-02-25/scan105/train/ours_30000/  \
     --mask_dir /home/borth/2d-gaussian-splatting/data/DTU \
     --DTU /home/borth/2d-gaussian-splatting/data/Offical_DTU_Dataset
```