import logging
import os

import hydra
import open3d as o3d
from omegaconf import DictConfig

from lib.gaussian_renderer import GaussianModel, Renderer
from lib.scene import Scene
from lib.utils.mesh_utils import GaussianExtractor, post_process_mesh
from lib.utils.render_utils import create_videos, generate_path

log = logging.getLogger()


@hydra.main(version_base=None, config_path="./conf", config_name="train")
def main(cfg: DictConfig):
    log.info("==> initializing configs ...")
    dataset = cfg.dataset
    pipe = cfg.pipeline
    iteration = cfg.mesh.iteration
    model_path = cfg.dataset.model_path

    log.info("==> initializing model ...")
    gaussians = GaussianModel(dataset.sh_degree)

    log.info("==> initializing dataset ...")
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    log.info("==> initializing renderer ...")
    render = Renderer(pipe=pipe, dataset=dataset)

    log.info("==> initializing extractor ...")
    train_dir = os.path.join(model_path, f"train/ours_{scene.loaded_iter}")
    test_dir = os.path.join(model_path, f"test/ours_{scene.loaded_iter}")
    traj_dir = os.path.join(model_path, f"traj/ours_{scene.loaded_iter}")
    extractor = GaussianExtractor(gaussians, render)

    if not cfg.mesh.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        extractor.reconstruction(scene.getTrainCameras())
        extractor.export_image(train_dir)

    if (not cfg.mesh.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        extractor.reconstruction(scene.getTestCameras())
        extractor.export_image(test_dir)

    if cfg.mesh.render_path:
        print("render videos ...")
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        extractor.reconstruction(cam_traj)
        extractor.export_image(traj_dir)
        create_videos(
            base_dir=traj_dir,
            input_dir=traj_dir,
            out_name="render_traj",
            num_frames=n_fames,
        )

    if not cfg.mesh.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)

        # set the active_sh to 0 to export only diffuse texture
        extractor.gaussians.active_sh_degree = 0
        extractor.reconstruction(scene.getTrainCameras())

        # extract the mesh and save
        if cfg.mesh.unbounded:
            name = "fuse_unbounded.ply"
            mesh = extractor.extract_mesh_unbounded(resolution=cfg.mesh.resolution)
        else:
            name = "fuse.ply"
            depth_trunc = (
                (extractor.radius * 2.0)
                if cfg.mesh.depth_trunc < 0
                else cfg.mesh.depth_trunc
            )
            voxel_size = (
                (depth_trunc / cfg.mesh.resolution)
                if cfg.mesh.voxel_size < 0
                else cfg.mesh.voxel_size
            )
            sdf_trunc = (
                5.0 * voxel_size if cfg.mesh.sdf_trunc < 0 else cfg.mesh.sdf_trunc
            )
            mesh = extractor.extract_mesh_bounded(
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc,
            )

        # raw mesh and save
        path = os.path.join(train_dir, name)
        o3d.io.write_triangle_mesh(path, mesh)
        print(f"mesh saved at {path}")

        # post-process the mesh and save, saving the largest N clusters
        path = os.path.join(train_dir, name.replace(".ply", "_post.ply"))
        mesh_post = post_process_mesh(mesh, cluster_to_keep=cfg.mesh.num_clusters)
        o3d.io.write_triangle_mesh(path, mesh_post)
        print(f"mesh post processed saved at {path}")


if __name__ == "__main__":
    main()
