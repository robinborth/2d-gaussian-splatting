import logging
from random import randint

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from lib.gaussian_renderer import render
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

    log.info(f"==> initializing logger <{cfg.logger._target_}> ...")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("==> initializing dataset...")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    log.info("==> initializing model ...")
    gaussians.training_setup(opt)
    if cfg.checkpoint:
        (model_params, first_iter) = torch.load(cfg.checkpoint)
        gaussians.restore(model_params, opt)

    log.info("==> setup optimization ...")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss = 0.0
    ema_dist_loss = 0.0
    ema_normal_loss = 0.0

    log.info("==> start optimization ...")
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # render current state
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss_l1 = l1_loss(image, gt_image)
        loss_ssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - loss_ssim)

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()

        iter_end.record()

        # logging
        with torch.no_grad():
            # Progress bar
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            ema_dist_loss = 0.4 * dist_loss.item() + 0.6 * ema_dist_loss
            ema_normal_loss = 0.4 * normal_loss.item() + 0.6 * ema_normal_loss

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss:.{5}f}",
                    "distort": f"{ema_dist_loss:.{5}f}",
                    "normal": f"{ema_normal_loss:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            metrics = {
                "train/total_loss": loss.item(),
                "train/dist_loss": ema_dist_loss,
                "train/normal_loss": ema_normal_loss,
                "train/reg_loss": loss_l1.item(),
                "train/iter_time": iter_start.elapsed_time(iter_end),
                "train/total_poitns": scene.gaussians.get_xyz.shape[0],
            }
            logger.log_metrics(metrics=metrics, step=iteration)

            # training_report(
            #     tb_writer,
            #     iteration,
            #     Ll1,
            #     loss,
            #     l1_loss,
            #     iter_start.elapsed_time(iter_end),
            #     testing_iterations,
            #     scene,
            #     render,
            #     (pipe, background),
            # )
            if iteration in cfg.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < cfg.optimization.iterations:
                gaussians.optimizer.step()  # type: ignore
                gaussians.optimizer.zero_grad(set_to_none=True)  # type: ignore

            if iteration in cfg.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


if __name__ == "__main__":
    main()
