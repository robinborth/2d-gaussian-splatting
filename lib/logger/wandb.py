import os
from pathlib import Path
from typing import Any

import open3d as o3d
import torch
from lightning.pytorch.loggers.wandb import WandbLogger

from lib.utils.eval_utils import evaluate, mesh_to_pcd
from lib.utils.general_utils import colormap
from lib.utils.image_utils import psnr
from lib.utils.loss_utils import l1_loss
from lib.utils.mesh_utils import GaussianExtractor, cull_scan_dtu, post_process_mesh


class GaussianLogger(WandbLogger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ema_loss = 0.0
        self.ema_dist_loss = 0.0
        self.ema_normal_loss = 0.0

    @torch.no_grad()
    def progress_step(
        self,
        loss,
        dist_loss,
        normal_loss,
        gaussians,
        iteration,
        iterations,
        progress_bar,
    ):
        # Progress bar
        self.ema_loss = 0.4 * loss.item() + 0.6 * self.ema_loss
        self.ema_dist_loss = 0.4 * dist_loss.item() + 0.6 * self.ema_dist_loss
        self.ema_normal_loss = 0.4 * normal_loss.item() + 0.6 * self.ema_normal_loss

        if iteration % 10 == 0:
            loss_dict = {
                "Loss": f"{self.ema_loss:.{5}f}",
                "distort": f"{self.ema_dist_loss:.{5}f}",
                "normal": f"{self.ema_normal_loss:.{5}f}",
                "Points": f"{len(gaussians.get_xyz)}",
            }
            progress_bar.set_postfix(loss_dict)
            progress_bar.update(10)

        if iteration == iterations:
            progress_bar.close()

    @torch.no_grad()
    def report(self, scene, render):
        torch.cuda.empty_cache()

        # Report test and samples of training set
        for config in scene.get_validation_configs(reduced=True):
            l1_test = 0.0
            psnr_test = 0.0
            for idx, camera in enumerate(config["cameras"]):
                suffix = f"{config['name']}_view_{camera.image_name}"
                I = render(camera, scene.gaussians)
                image = torch.clamp(I.render, 0.0, 1.0).to("cuda")
                gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                if idx < 5:
                    # postprocess the image information
                    rend_normal = I.rend_normal * 0.5 + 0.5
                    surf_normal = I.surf_normal * 0.5 + 0.5
                    depth = I.surf_depth / I.surf_depth.max()
                    depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                    rend_dist = colormap(I.rend_dist.cpu().numpy()[0])
                    # log the image information
                    self.log_image(f"{suffix}/depth", [depth[None]])
                    self.log_image(f"{suffix}/render", [image[None]])
                    self.log_image(f"{suffix}/rend_normal", [rend_normal[None]])
                    self.log_image(f"{suffix}/surf_normal", [surf_normal[None]])
                    self.log_image(f"{suffix}/rend_alpha", [I.rend_alpha[None]])
                    self.log_image(f"{suffix}/rend_dist", [rend_dist[None]])
                    self.log_image(f"{suffix}/ground_truth", [gt_image[None]])
                # perform the metric computation
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            # log the metric computation
            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            metrics = {
                f"{config['name']}/viewpoint/l1_loss": l1_test,
                f"{config['name']}/viewpoint/psnr": psnr_test,
            }
            self.log_metrics(metrics=metrics)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def mesh(
        self,
        scene,
        render,
        iteration: int,
        voxel_size=0.004,
        sdf_trunc=0.02,
        depth_trunc=3,
        num_clusters=50,
        fuse_post: bool = True,
        fuse_cull: bool = True,
    ):
        for config in scene.get_validation_configs(reduced=False):
            output_dir = Path(str(self.save_dir)) / f"{config['name']}/ours_{iteration}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # create the cameras
            extractor = GaussianExtractor(scene.gaussians, render)
            sh_degree = extractor.gaussians.active_sh_degree
            extractor.gaussians.active_sh_degree = 0
            extractor.reconstruction(viewpoint_stack=config["cameras"])
            extractor.gaussians.active_sh_degree = sh_degree

            # create the mesh
            mesh = extractor.extract_mesh_bounded(
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc,
            )
            mesh_path = str(output_dir / "fuse.ply")
            o3d.io.write_triangle_mesh(mesh_path, mesh)

            # post process the mesh
            if fuse_post:
                mesh_post = post_process_mesh(mesh, cluster_to_keep=num_clusters)
                mesh_path = str(output_dir / "fuse_post.ply")
                o3d.io.write_triangle_mesh(mesh_path, mesh_post)

            # cull mesh based on the mask
            if fuse_cull:
                cull_scan_dtu(source_path=scene.source_path, mesh_path=mesh_path)

    @torch.no_grad()
    def evaluate(
        self,
        scene,
        scan_id,
        dataset_dir,  # DTU-Dataset Original
        iteration,
        mesh_name,
        patch_size,
        max_dist,
        downsample_density,
    ):
        for config in scene.get_validation_configs(reduced=False):
            output_dir = Path(str(self.save_dir)) / f"{config['name']}/ours_{iteration}"
            mesh_path = str(output_dir / mesh_name)
            data_pcd = mesh_to_pcd(mesh_path=mesh_path, thresh=downsample_density)
            metrics = evaluate(
                data_pcd=data_pcd,
                scan_id=scan_id,
                dataset_dir=dataset_dir,
                patch_size=patch_size,
                max_dist=max_dist,
                downsample_density=downsample_density,
            )
            self.log_metrics({f"{config['name']}/{k}": v for k, v in metrics.items()})
