import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import render_utils as rend_util
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm

from lib.utils.mesh_utils import cull_scan_dtu

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments to evaluate the mesh.")

    parser.add_argument(
        "--input_mesh", type=str, help="path to the mesh to be evaluated"
    )
    parser.add_argument("--scan_id", type=str, help="scan id of the input mesh")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results_single",
        help="path to the output folder",
    )
    parser.add_argument(
        "--mask_dir", type=str, default="mask", help="path to uncropped mask"
    )
    parser.add_argument(
        "--DTU",
        type=str,
        default="Offical_DTU_Dataset",
        help="path to the GT DTU point clouds",
    )
    args = parser.parse_args()

    Offical_DTU_Dataset = args.DTU
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scan = args.scan_id
    ply_file = args.input_mesh
    print("cull mesh ....")
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
    cull_scan_dtu(
        source_path=os.path.join(args.mask_dir, f"scan{args.scan_id}"),
        mesh_path=ply_file,
        mesh_name="culled_mesh.ply",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f"python {script_dir}/eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
    os.system(cmd)
