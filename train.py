import logging
from random import randint

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from lib.gaussian_renderer import Renderer
from lib.scene import GaussianModel, Scene
from lib.utils.general_utils import safe_state
from lib.utils.loss_utils import l1_loss, ssim

log = logging.getLogger()


@hydra.main(version_base=None, config_path="./conf", config_name="train")
def main(cfg: DictConfig):
    log.info("==> initializing configs ...")
    # initialize system state
    first_iter = 0
    safe_state(cfg.quiet)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    # extract the param groups
    dataset = cfg.dataset
    opt = cfg.optimization
    pipe = cfg.pipeline

    log.info("==> initializing renderer ...")
    render = Renderer(pipe=pipe, dataset=dataset)

    log.info(f"==> initializing logger <{cfg.logger._target_}> ...")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("==> initializing dataset ...")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    log.info("==> initializing model ...")
    gaussians.training_setup(opt)
    if cfg.checkpoint:
        (model_params, first_iter) = torch.load(cfg.checkpoint)
        gaussians.restore(model_params, opt)

    log.info("==> setup optimization ...")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    cameras = None

    log.info("==> start optimization ...")
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        # init iteration step
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # pick a random camera from the remaining
        if not cameras:
            cameras = scene.getTrainCameras().copy()
        camera = cameras.pop(randint(0, len(cameras) - 1))

        # render current state
        I = render(camera, gaussians)

        # compute photometic loss
        gt_image = camera.original_image.cuda()
        loss_l1 = l1_loss(I.render, gt_image)
        loss_ssim = ssim(I.render, gt_image)
        loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - loss_ssim)

        # compute regularization loss
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        normal_error = (1 - (I.rend_normal * I.surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (I.rend_dist).mean()

        # compute total loss and perform backward pass
        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        iter_end.record()

        # update the progress bar
        logger.progress_step(
            loss=loss,
            dist_loss=dist_loss,
            normal_loss=normal_loss,
            gaussians=gaussians,
            iteration=iteration,
            iterations=opt.iterations,
            progress_bar=progress_bar,
        )

        # update the metrics
        metrics = {
            "train/total_loss": loss.item(),
            "train/dist_loss": logger.ema_dist_loss,
            "train/normal_loss": logger.ema_normal_loss,
            "train/reg_loss": loss_l1.item(),
            "train/iter_time": iter_start.elapsed_time(iter_end),
            "train/total_poitns": scene.gaussians.get_xyz.shape[0],
        }
        logger.log_metrics(metrics=metrics, step=iteration)

        # perform detailed log
        if iteration in cfg.test_iterations:
            logger.report(scene=scene, render=render)
            logger.mesh(
                scene=scene,
                render=render,
                iteration=iteration,
                voxel_size=cfg.mesh.voxel_size,
                sdf_trunc=cfg.mesh.sdf_trunc,
                depth_trunc=cfg.mesh.depth_trunc,
                num_clusters=cfg.mesh.num_clusters,
                fuse_post=cfg.mesh.fuse_post,
                fuse_cull=cfg.mesh.fuse_cull,
            )
            logger.evaluate(
                scene=scene,
                iteration=iteration,
                scan_id=cfg.eval.scan,
                dataset_dir=cfg.eval.dataset_dir,
                mesh_name=cfg.eval.mesh,
                patch_size=cfg.eval.path_size,
                max_dist=cfg.eval.max_dist,
                downsample_density=cfg.eval.downsample_density,
            )

        if iteration in cfg.save_iterations:
            log.info(f"[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)

        # perform densification
        gaussians.densification(
            I=I,
            iteration=iteration,
            opt=opt,
            dataset=dataset,
            scene=scene,
        )

        # perform optimizer step
        if iteration < cfg.optimization.iterations:
            gaussians.optimizer.step()  # type: ignore
            gaussians.optimizer.zero_grad(set_to_none=True)  # type: ignore

        # save the gaussian checkpoint
        if iteration in cfg.checkpoint_iterations:
            log.info(f"[ITER {iteration}] Saving Checkpoint")
            path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
            torch.save((gaussians.capture(), iteration), path)


if __name__ == "__main__":
    main()
