import random
from functools import partial

import numpy as np
import open3d as o3d
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    CamerasBase,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

################################################################################
# Utilities
################################################################################


def compute_half_pixel_princial_point(image_size: int):
    cx, cy = (image_size - 1) / 2, (image_size - 1) / 2
    return cx, cy


def compute_focal_length_from_FoV(camera: CamerasBase, image_size: int):
    focal = image_size / (2 * torch.tan(torch.deg2rad(camera.fov[0] / 2)))
    return focal, focal


def get_depth_camera_space_attributes(mesh: Meshes, camera: CamerasBase):
    P = camera.get_world_to_view_transform()
    v_camera = P.transform_points(mesh.verts_packed())
    return v_camera[None][..., 2:]  # (B,V,D)


def get_points_camera_space_attributes(mesh: Meshes, camera: CamerasBase):
    P = camera.get_world_to_view_transform()
    v_camera = P.transform_points(mesh.verts_packed())
    return v_camera[None]  # (B,V,D)


def select_random_points_and_normals(
    points: torch.Tensor,
    normals: torch.Tensor,
    max_samples: int,
):
    assert points.shape == normals.shape
    idx = torch.randperm(len(points))[:max_samples]
    return points[idx], normals[idx]


def select_random_points(points: torch.Tensor, max_samples: int):
    return points[torch.randperm(len(points))[:max_samples]]


def select_random_camera(cameras: list):
    idx = random.choice(range(len(cameras)))
    return cameras[idx], idx


def load_mesh(path: str, device: str = "cuda"):
    return load_objs_as_meshes([path], device=device)


################################################################################
# Surface Processing
################################################################################


def depth_map_to_points_camera_space(
    depth: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    padding: bool = False,
):
    if padding:
        zfar = depth.max().item()
        depth = torch.nn.functional.pad(depth, (1, 1, 1, 1), value=zfar)
        cx += 1
        cy += 1

    # we assume that the input image width and height are the same
    B, H, W = depth.shape
    assert H == W

    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing="ij",
    )
    # compute normalized pixel coordinates
    x = (x - cx) / fx
    y = (y - cy) / fy

    # compute 3D points
    X = x * depth
    Y = y * depth
    Z = depth

    return torch.stack([-X, -Y, Z], dim=-1).reshape(B, H, W, 3)


def depth_to_points(
    depth: torch.Tensor,
    mask: torch.Tensor,
    camera: CamerasBase,
    padding: bool = False,
    fill_depth: str = "zfar",  # zfar, max, max(d)
):
    # prepare the correct fill value
    assert fill_depth in ["zfar", "max", "max2", "max5", "max10"]
    fill_value = camera.zfar
    if fill_depth == "max":
        fill_value = depth.max()
    elif fill_depth.startswith("max"):
        multiplyer = int(fill_depth.split("max")[-1])
        fill_value = depth.max() * multiplyer

    # extract the shape and make sure that it follow the format
    _, H, W, _ = depth.shape
    assert H == W
    fx, fy = compute_focal_length_from_FoV(camera=camera, image_size=W)
    cx, cy = compute_half_pixel_princial_point(image_size=W)
    # the point cloud has dim (B, H+2, W+2, 3)
    depth_map = depth[..., 0]
    depth_map[mask] = fill_value

    return depth_map_to_points_camera_space(depth_map, fx, fy, cx, cy, padding)


def depth_to_normals(
    depth: torch.Tensor,
    mask: torch.Tensor,
    camera: CamerasBase,
    fill_depth: str = "zfar",
):
    pcd = depth_to_points(
        depth=depth,
        mask=mask,
        camera=camera,
        padding=True,
        fill_depth=fill_depth,
    )
    N_x = pcd[:, :, 2:, :] - pcd[:, :, :-2, :]
    N_y = pcd[:, 2:, :, :] - pcd[:, :-2, :, :]
    normal = torch.linalg.cross(N_y[:, :, 1:-1, :], N_x[:, 1:-1, :, :])
    normal /= torch.linalg.vector_norm(normal, dim=-1)[..., None]
    return normal


def extract_surface_data(
    camera: CamerasBase,
    mesh: Meshes,
    image_size: int,
    fill_depth="zfar",
):
    attributes = get_depth_camera_space_attributes(mesh=mesh, camera=camera)
    depth_map, mask = rasterize_attributes(
        mesh=mesh,
        camera=camera,
        attributes=attributes,
        image_size=image_size,
    )
    normal_map = depth_to_normals(
        depth=depth_map,
        mask=mask,
        camera=camera,
        fill_depth=fill_depth,
    )
    point_map = depth_to_points(
        depth=depth_map,
        mask=mask,
        camera=camera,
        fill_depth=fill_depth,
    )

    # remove the batch_size
    normal_map = normal_map[0]
    point_map = point_map[0]
    depth_map = depth_map[0]
    mask = mask[0]

    # transform point and normal map to world space
    P = camera.get_world_to_view_transform()
    normal_map = P.inverse().transform_normals(normal_map)
    point_map = P.inverse().transform_points(point_map)

    # compute the points in world space and remove the masks
    normals = normal_map[~mask]
    points = point_map[~mask]

    # compute the indicator map
    indicator_map = torch.ones_like(mask)
    indicator_map[mask] = 0.0

    return {
        "mask": mask,
        "indicator_map": indicator_map,
        "normal_map": normal_map,  # in world space
        "point_map": point_map,  # in world space
        "normals": normals,
        "points": points,
    }


def extract_points_data(
    # dataset settings
    cameras: CamerasBase,
    mesh: Meshes,
    image_size: int,
    fill_depth="zfar",
    # empty space sampling
    empty_points_per_ray: int = 4,
    close_points_per_ray: int = 2,
    close_points_surface_threshold: float = 0.01,
):
    normals = []
    points_surface = []
    points_close = []
    points_empty = []
    indicator_maps = []
    normal_maps = []
    point_maps = []
    masks = []

    for camera in cameras:
        # extract the surface data
        data = extract_surface_data(
            camera=camera,
            mesh=mesh,
            image_size=image_size,
            fill_depth=fill_depth,
        )
        indicator_maps.append(data["indicator_map"])
        normal_maps.append(data["normal_map"])
        point_maps.append(data["point_map"])
        masks.append(data["mask"])

        # extract the points data
        normals.append(data["normals"])
        points_surface.append(data["points"])

        # extract the close surface points
        _points = sample_empty_space_points(
            points=data["points"],
            camera=camera,
            samples=close_points_per_ray,
            surface_threshold=close_points_surface_threshold,
        )
        points_close.append(_points)

        # extract the empty space points
        points = sample_empty_space_points(
            points=data["points"],
            camera=camera,
            samples=empty_points_per_ray,
            surface_threshold=1.0,
        )
        points_empty.append(points)

    # merge the information together
    return {
        "points_surface": torch.cat(points_surface),  # (P, 3)
        "points_empty": torch.cat(points_empty),  # (P, 3)
        "points_close": torch.cat(points_close),  # (P, 3)
        "normals": torch.cat(normals),  # (P, 3)
        "indicator_maps": indicator_maps,  # (B, H, W, 3)
        "normal_maps": normal_maps,  # (B, H, W, 3)
        "point_maps": point_maps,  # (B, H, W, 3)
        "masks": masks,  # (B, H, W, 3)
    }


################################################################################
# Camera Utilties
################################################################################


def uniform_sphere_cameras(dist: float = 1.0, segments: int = 10, device: str = "cuda"):
    cameras = []
    elevs = torch.linspace(0, 360, segments + 1)[:segments]
    azims = torch.linspace(0, 360, segments + 1)[:segments]
    for elev in elevs:
        for azim in azims:
            R, T = look_at_view_transform(dist, elev, azim)
            camera = FoVPerspectiveCameras(device=device, R=R, T=T)
            cameras.append(camera)
    return cameras


def barycentric_interpolation(
    vertices_idx: torch.Tensor,
    bary_coords: torch.Tensor,
    attributes: torch.Tensor,
):
    # access the vertex attributes
    B, H, W, _ = vertices_idx.shape  # (B, H, W, 3)
    _, _, D = attributes.shape  # (B, V, D)

    # Flatten the vertices_idx and bary_coords to reduce unnecessary operations
    flat_vertices_idx = vertices_idx.view(B, -1)  # (B, H*W*3)
    flat_vertices_idx = flat_vertices_idx.unsqueeze(-1).expand(-1, -1, D)
    # Efficiently gather the vertex attributes in one step
    vertex_attributes = attributes.gather(1, flat_vertices_idx)  # (B, H*W*3, D)

    # Reshape gathered attributes to (B, H, W, 3, D) directly
    vertex_attributes = vertex_attributes.view(B, H, W, 3, D)

    # Perform the weighted sum using barycentric coordinates
    bary_coords = bary_coords.unsqueeze(-1)  # (B, H, W, 3, 1)
    attributes = (bary_coords * vertex_attributes).sum(dim=-2)  # (B, H, W, D)

    return attributes


def rasterize_attributes(
    mesh: Meshes,
    camera: CamerasBase,
    attributes: torch.Tensor,
    image_size: int = 512,
    mask: bool = True,
    fill_value: float = 0.0,
):
    # define the rasterizer
    raster_settings = RasterizationSettings(image_size=image_size)
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    # render the image
    fragments = rasterizer(mesh)

    # select closest face (B,H,W,K,3) -> (B,H,W,3)
    bary_coords = fragments.bary_coords[..., 0, :]
    # compute the vertices idx
    faces = mesh.faces_packed()  # (F,3)
    # select closest face idx to the view (B,H,W,K) -> (B,H,W)
    face_idx = fragments.pix_to_face[..., 0]
    # select the vertex idx (B,H,W,3)
    vertices_idx = faces[face_idx]

    # interpolate the barycentric coordinates
    attributes = barycentric_interpolation(
        vertices_idx=vertices_idx,
        bary_coords=bary_coords,
        attributes=attributes,
    )

    # the mask of invalid pixels with an intersection
    mask_idx = face_idx == -1
    if mask:
        attributes[mask_idx] = fill_value

    return attributes, mask_idx


################################################################################
# Point Processing Utilties
################################################################################


def sample_empty_space_points(
    points: torch.Tensor,
    camera: CamerasBase,
    samples: int = 6,
    surface_threshold: float = 1.0,
):
    p_s = torch.repeat_interleave(points, samples, dim=0)  # (P*samples, 3)
    s = camera.get_camera_center()  # (1, 3)
    t = torch.rand((p_s.shape[0], 1), device=p_s.device)
    t = 1 - t * surface_threshold  # ensures that the samples are close to the surface
    p_e = s + t * (p_s - s)  # (P_e, 3)
    return p_e


def subsample_points(points: torch.Tensor, resolution: float = 0.01, normals=None):
    # fill the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())

    # downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=resolution)

    # extract the points
    points = torch.tensor(np.asarray(pcd.points)).to(points)
    if normals is None:
        return points

    # extract the normals if normals are provided as input
    normals = torch.tensor(np.asarray(pcd.normals)).to(normals)
    return points, normals


def subsample_dataset_points(
    points_surface: torch.Tensor,
    points_empty: torch.Tensor,
    points_close: torch.Tensor,
    normals: torch.Tensor,
    # subsampling settings
    resolution: float = 0.01,
    empty_space_max_ratio: float = -1.0,
    **kwargs,
):
    # subsample points to the desired resolution
    points_surface, normals = subsample_points(
        points=points_surface,
        normals=normals,
        resolution=resolution,
    )
    points_close = subsample_points(points=points_close, resolution=resolution)

    # subsample empty points and ensure similar ratio
    points_empty = subsample_points(points=points_empty, resolution=resolution)
    if empty_space_max_ratio > 0:
        surface_count = points_surface.shape[0] + points_close.shape[0]
        max_empty_points = int(surface_count * empty_space_max_ratio)
        max_empty_points = min(max_empty_points, points_empty.shape[0])
        indices = torch.randperm(points_empty.shape[0])[:max_empty_points]
        points_empty = points_empty[indices]

    # return all the information
    return points_surface, points_close, points_empty, normals


################################################################################
# Different Approaches to Estimate the Vector Field
################################################################################


def estimate_vector_field_cluster(
    points: torch.Tensor,
    normals: torch.Tensor,
    query: torch.Tensor,
    k: int = 20,
    sigma: float = 1.0,
    threshold: float = 30.0,
    normalize: bool = True,
    chunk_size: int = 1_000,
):
    vectors = []
    for q in torch.split(query, chunk_size):
        # compute the distances
        distances = torch.cdist(q, points, p=2)
        distances, indices = torch.topk(distances, k, dim=1, largest=False)

        # gaussian-weighted average of the k nearest neighbors
        weights = torch.exp(-distances / (2 * sigma))

        # compute the clusters
        cluster_idxs = torch.full_like(indices, -1)
        cluster_idxs[:, 0] = 0
        for i in range(1, k):
            prev_idxs = cluster_idxs[:, :i]
            prev_max_cluster = prev_idxs.max(dim=-1).values

            # default is just the next index
            current_normal = normals[indices][:, i, :]
            current_idxs = prev_max_cluster + 1
            tmp_cluster_similarity = -torch.ones(q.shape[0]).to(q)  # fill with -1

            # compute the previous cluster vectors
            for j in range(0, i):
                # activate the current cluster by disabeling all others
                cluster_weights = weights.clone()
                cluster_weights[cluster_idxs != j] = 0.0
                cluster_vector = normals[indices] * cluster_weights.unsqueeze(-1)
                cluster_vector = cluster_vector.sum(-2)
                cluster_vector /= cluster_weights.sum(-1, keepdim=True)

                # compute cosine similartiy between cluster vector and current vector
                similarity = (cluster_vector * current_normal).sum(-1)
                similarity /= torch.linalg.vector_norm(cluster_vector, dim=-1)
                similarity /= torch.linalg.vector_norm(current_normal, dim=-1)

                # compute the angle
                theta = torch.rad2deg(torch.acos(similarity))

                # update the best cluster
                mask = (similarity > tmp_cluster_similarity) & (theta <= threshold)
                tmp_cluster_similarity[mask] = similarity[mask]
                current_idxs[mask] = j

            # if there is a cluster that matches
            cluster_idxs[:, i] = current_idxs

        # evalute the cluster normals and centers
        _cluster_vectors = []
        _cluster_centers = []
        for j in range(0, k):
            # activate the current cluster by disabeling all others
            cluster_weights = weights.clone()
            cluster_weights[cluster_idxs != j] = 0.0
            cluster_vector = (normals[indices] * cluster_weights.unsqueeze(-1)).sum(-2)
            cluster_vector /= cluster_weights.sum(-1, keepdim=True)
            _cluster_vectors.append(cluster_vector)
            # activate the current cluster centers
            cluster_center = (points[indices] * cluster_weights.unsqueeze(-1)).sum(-2)
            _cluster_centers.append(cluster_center)
        cluster_centers = torch.stack(_cluster_centers, dim=1)  # (P,C,3)
        cluster_vectors = torch.stack(_cluster_vectors, dim=1)  # (P,C,3)

        # select the cluster normal with the clostest cluster center
        distances = q.unsqueeze(-2) - cluster_centers
        distances = torch.linalg.vector_norm(distances, dim=-1)  # (Q, C)
        # reset the distances with no values
        idxs = (
            torch.arange(distances.shape[1])
            .expand(distances.shape[0], -1)
            .to(distances)
        )
        mask = idxs > (cluster_idxs.max(dim=-1).values)[..., None]
        distances[mask] = torch.nan
        distances, indices = torch.topk(distances, 1, dim=1, largest=False)
        indices = indices[..., 0]  # just the top 1
        # the final cluster vectors
        vector = cluster_vectors[torch.arange(cluster_vectors.shape[0]), indices]

        # normalize the vector field to contain only normal vectors
        if normalize:
            vector /= torch.linalg.vector_norm(vector, dim=-1).unsqueeze(-1)
        vectors.append(vector)
    # return the inverse of the normal as the vector field
    return -torch.cat(vectors)


def estimate_vector_field_k_nearest_neighbors(
    points: torch.Tensor,
    normals: torch.Tensor,
    query: torch.Tensor,
    k: int = 20,
    sigma: float = 1.0,
    normalize: bool = True,
    chunk_size: int = 1_000,
):
    vectors = []
    for q in torch.split(query, chunk_size):
        # compute the distances
        distances = torch.cdist(q, points, p=2)
        distances, indices = torch.topk(distances, k, dim=1, largest=False)

        # gaussian-weighted average of the k nearest neighbors
        weights = torch.exp(-distances / (2 * sigma))
        normalization = weights.sum(-1, keepdim=True)
        vector = (normals[indices] * weights.unsqueeze(-1)).sum(-2) / normalization

        # normalize the vector field to contain only normal vectors
        if normalize:
            vector /= torch.linalg.vector_norm(vector, dim=-1).unsqueeze(-1)
        vectors.append(vector)
    # return the inverse of the normal as the vector field
    return -torch.cat(vectors)


def estimate_vector_field_nearest_neighbor(
    points: torch.Tensor,
    normals: torch.Tensor,
    query: torch.Tensor,
    normalize: bool = True,
    chunk_size: int = 1_000,
):
    vectors = []
    for q in torch.split(query, chunk_size):
        # compute the distances
        distances = torch.cdist(q, points, p=2)
        distances, indices = torch.topk(distances, 1, dim=1, largest=False)
        vector = normals[indices[..., 0]]
        # normalize the vector field to contain only normal vectors
        if normalize:
            vector /= torch.linalg.vector_norm(vector, dim=-1).unsqueeze(-1)
        vectors.append(vector)
    # return the inverse of the normal as the vector field
    return -torch.cat(vectors)


def select_vector_field_function(
    vector_field_mode: str = "nearest_neighbor",
    normalize: bool = True,
    chunk_size: int = 1_000,
    k: int = 20,
    sigma: float = 1.0,
    threshold: float = 30,
):
    if vector_field_mode == "nearest_neighbor":
        return partial(
            estimate_vector_field_nearest_neighbor,
            normalize=normalize,
            chunk_size=chunk_size,
        )
    if vector_field_mode == "k_nearest_neighbors":
        return partial(
            estimate_vector_field_k_nearest_neighbors,
            k=k,
            sigma=sigma,
            normalize=normalize,
            chunk_size=chunk_size,
        )
    if vector_field_mode == "cluster":
        return partial(
            estimate_vector_field_cluster,
            k=k,
            sigma=sigma,
            normalize=normalize,
            chunk_size=chunk_size,
            threshold=threshold,
        )
    raise AttributeError(f"Please select a correct {vector_field_mode=}!")
