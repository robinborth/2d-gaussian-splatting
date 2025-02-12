import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.arguments import ModelParams, OptimizationParams, PipelineParams
from lib.gaussian_renderer import network_gui, render
from lib.scene import GaussianModel, Scene
from lib.utils.general_utils import safe_state
from lib.utils.image_utils import psnr, render_net_image
from lib.utils.loss_utils import l1_loss, ssim


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if iteration not in testing_iterations:
        return

    # Report test and samples of training set
    torch.cuda.empty_cache()
    for config in scene.get_validation_configs():
        l1_test = 0.0
        psnr_test = 0.0
        for idx, viewpoint in enumerate(config["cameras"]):
            render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if idx < 5:
                from lib.utils.general_utils import colormap

                depth = render_pkg["surf_depth"]
                norm = depth.max()
                depth = depth / norm
                depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                tb_writer.add_images(
                    config["name"] + "_view_{}/depth".format(viewpoint.image_name),
                    depth[None],
                    global_step=iteration,
                )
                tb_writer.add_images(
                    config["name"] + "_view_{}/render".format(viewpoint.image_name),
                    image[None],
                    global_step=iteration,
                )

                try:
                    rend_alpha = render_pkg["rend_alpha"]
                    rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                    surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                    tb_writer.add_images(
                        config["name"]
                        + "_view_{}/rend_normal".format(viewpoint.image_name),
                        rend_normal[None],
                        global_step=iteration,
                    )
                    tb_writer.add_images(
                        config["name"]
                        + "_view_{}/surf_normal".format(viewpoint.image_name),
                        surf_normal[None],
                        global_step=iteration,
                    )
                    tb_writer.add_images(
                        config["name"]
                        + "_view_{}/rend_alpha".format(viewpoint.image_name),
                        rend_alpha[None],
                        global_step=iteration,
                    )

                    rend_dist = render_pkg["rend_dist"]
                    rend_dist = colormap(rend_dist.cpu().numpy()[0])
                    tb_writer.add_images(
                        config["name"]
                        + "_view_{}/rend_dist".format(viewpoint.image_name),
                        rend_dist[None],
                        global_step=iteration,
                    )
                except:
                    pass

                if iteration == testing_iterations[0]:
                    tb_writer.add_images(
                        config["name"]
                        + "_view_{}/ground_truth".format(viewpoint.image_name),
                        gt_image[None],
                        global_step=iteration,
                    )

            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(config["cameras"])
        l1_test /= len(config["cameras"])
        print(
            "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                iteration, config["name"], l1_test, psnr_test
            )
        )

        metrics = {
            f"{config['name']}/viewpoint/l1_loss": l1_test,
            f"{config['name']}/viewpoint/psnr": psnr_test,
        }
        logger.log_metrics(metrics=metrics, step=iteration)

    torch.cuda.empty_cache()
