# adapted from https://github.com/jzhangbs/DTUeval-python
import argparse
import multiprocessing as mp

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from scipy.io import loadmat


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[: n1 + 1, : n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def mesh_to_pcd(mesh_path, thresh):
    # read data mesh
    data_mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    # sample pcd from mesh
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(
            sample_single_tri,
            (
                (
                    n1[i, 0],
                    n2[i, 0],
                    v1[i : i + 1],
                    v2[i : i + 1],
                    tri_vert[i : i + 1, 0],
                )
                for i in range(len(n1))
            ),
            chunksize=1024,
        )

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)
    return data_pcd


def evaluate(
    data_pcd,
    scan_id: int,
    dataset_dir: str,
    patch_size: int,
    max_dist: int,
    downsample_density: int,
):
    # random shuffle pcd index
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    # downsample pcd
    nn_engine = skln.NearestNeighbors(
        n_neighbors=1,
        radius=downsample_density,
        algorithm="kd_tree",
        n_jobs=-1,
    )
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(
        data_pcd, radius=downsample_density, return_distance=False
    )
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    # masking data pcd
    obs_mask_file = loadmat(f"{dataset_dir}/ObsMask/ObsMask{scan_id}_10.mat")
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ["ObsMask", "BB", "Res"]]
    BB = BB.astype(np.float32)

    patch = patch_size
    inbound = (data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)
    inbound = inbound.sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = (data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))
    grid_inbound = grid_inbound.sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]]
    in_obs = in_obs.astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    # read STL pcd
    path = f"{dataset_dir}/Points/stl/stl{scan_id:03}_total.ply"
    stl_pcd = o3d.io.read_point_cloud(path)
    stl = np.asarray(stl_pcd.points)

    # compute data2stl
    nn_engine.fit(stl)
    dist_d2s, _ = nn_engine.kneighbors(
        data_in_obs,
        n_neighbors=1,
        return_distance=True,
    )
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    # compute stl2data
    ground_plane = loadmat(f"{dataset_dir}/ObsMask/Plane{scan_id}.mat")["P"]
    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]
    nn_engine.fit(data_in)
    dist_s2d, _ = nn_engine.kneighbors(
        stl_above,
        n_neighbors=1,
        return_distance=True,
    )
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    # compute overall chamfer
    over_all = (mean_d2s + mean_s2d) / 2

    return {
        "mean_d2s": mean_d2s,
        "mean_s2d": mean_s2d,
        "overall": over_all,
    }
