# adapted from https://github.com/jzhangbs/DTUeval-python
import json
import multiprocessing as mp

import hydra
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from omegaconf import DictConfig
from scipy.io import loadmat
from tqdm import tqdm

from lib.utils.eval_utils import evaluate, mesh_to_pcd


@hydra.main(version_base=None, config_path="../../conf", config_name="optimize")
def main(cfg: DictConfig):
    mp.freeze_support()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, default="data_in.ply")
    # parser.add_argument("--scan", type=int, default=1)
    # parser.add_argument("--mode", type=str, default="mesh", choices=["mesh", "pcd"])
    # parser.add_argument("--dataset_dir", type=str, default=".")
    # parser.add_argument("--vis_out_dir", type=str, default=".")
    # parser.add_argument("--downsample_density", type=float, default=0.2)
    # parser.add_argument("--patch_size", type=float, default=60)
    # parser.add_argument("--max_dist", type=float, default=20)
    # parser.add_argument("--visualize_threshold", type=float, default=10)
    # cfg.eval = parser.parse_cfg.eval()
    pbar = tqdm(total=8)

    thresh = cfg.eval.downsample_density
    if cfg.eval.mode == "mesh":
        data_pcd = mesh_to_pcd(mesh_path=cfg.eval.data, thresh=thresh)
    elif cfg.eval.mode == "pcd":
        data_pcd_o3d = o3d.io.read_point_cloud(cfg.eval.data)
        data_pcd = np.asarray(data_pcd_o3d.points)

    pbar.update(1)
    pbar.set_description("random shuffle pcd index")
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description("downsample pcd")
    nn_engine = skln.NearestNeighbors(
        n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1
    )
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(
        data_pcd, radius=thresh, return_distance=False
    )
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description("masking data pcd")
    obs_mask_file = loadmat(
        f"{cfg.eval.dataset_dir}/ObsMask/ObsMask{cfg.eval.scan}_10.mat"
    )
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ["ObsMask", "BB", "Res"]]
    BB = BB.astype(np.float32)

    patch = cfg.eval.patch_size
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(
        axis=-1
    ) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = (
        (data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))
    ).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(
        np.bool_
    )
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description("read STL pcd")
    stl_pcd = o3d.io.read_point_cloud(
        f"{cfg.eval.dataset_dir}/Points/stl/stl{cfg.eval.scan:03}_total.ply"
    )
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description("compute data2stl")
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(
        data_in_obs, n_neighbors=1, return_distance=True
    )
    max_dist = cfg.eval.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description("compute stl2data")
    ground_plane = loadmat(f"{cfg.eval.dataset_dir}/ObsMask/Plane{cfg.eval.scan}.mat")[
        "P"
    ]

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(
        stl_above, n_neighbors=1, return_distance=True
    )
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    # pbar.update(1)
    # pbar.set_description("visualize error")
    # vis_dist = cfg.eval.visualize_threshold
    # R = np.array([[1, 0, 0]], dtype=np.float64)
    # G = np.array([[0, 1, 0]], dtype=np.float64)
    # B = np.array([[0, 0, 1]], dtype=np.float64)
    # W = np.array([[1, 1, 1]], dtype=np.float64)
    # data_color = np.tile(B, (data_down.shape[0], 1))
    # data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    # data_color[np.where(inbound)[0][grid_inbound][in_obs]] = R * data_alpha + W * (
    #     1 - data_alpha
    # )
    # data_color[
    #     np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:, 0] >= max_dist]
    # ] = G
    # write_vis_pcd(
    #     f"{cfg.eval.vis_out_dir}/vis_{cfg.eval.scan:03}_d2s.ply", data_down, data_color
    # )

    # stl_color = np.tile(B, (stl.shape[0], 1))
    # stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    # stl_color[np.where(above)[0]] = R * stl_alpha + W * (1 - stl_alpha)
    # stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_dist]] = G
    # write_vis_pcd(f"{cfg.eval.vis_out_dir}/vis_{cfg.eval.scan:03}_s2d.ply", stl, stl_color)

    # pbar.update(1)
    # pbar.set_description("done")
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)

    # with open(f"{cfg.eval.vis_out_dir}/results.json", "w") as fp:
    #     json.dump(metrics, fp, indent=True)


if __name__ == "__main__":
    main()
