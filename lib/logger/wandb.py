from typing import Any

import torch
from lightning.pytorch.loggers.wandb import WandbLogger

from lib.utils.general_utils import colormap
from lib.utils.image_utils import psnr
from lib.utils.loss_utils import l1_loss


class GaussianLogger(WandbLogger):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ema_loss = 0.0
        self.ema_dist_loss = 0.0
        self.ema_normal_loss = 0.0

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
        # logging
        with torch.no_grad():
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

    def report(
        self,
        scene,
        render,
        iteration,
        test_iterations,
    ):
        if iteration not in test_iterations:
            return

        # Report test and samples of training set
        torch.cuda.empty_cache()
        for config in scene.get_validation_configs():
            l1_test = 0.0
            psnr_test = 0.0
            for idx, camera in enumerate(config["cameras"]):
                I = render(camera, scene.gaussians)
                image = torch.clamp(I.render, 0.0, 1.0).to("cuda")
                gt_image = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                if idx < 5:

                    depth = I.surf_depth
                    norm = depth.max()
                    depth = depth / norm
                    depth = colormap(depth.cpu().numpy()[0], cmap="turbo")
                    self.log_image(
                        config["name"] + "_view_{}/depth".format(camera.image_name),
                        [depth[None]],
                        step=iteration,
                    )
                    self.log_image(
                        config["name"] + "_view_{}/render".format(camera.image_name),
                        [image[None]],
                        step=iteration,
                    )

                    rend_alpha = I.rend_alpha
                    rend_normal = I.rend_normal * 0.5 + 0.5
                    surf_normal = I.surf_normal * 0.5 + 0.5
                    self.log_image(
                        config["name"]
                        + "_view_{}/rend_normal".format(camera.image_name),
                        [rend_normal[None]],
                        step=iteration,
                    )
                    self.log_image(
                        config["name"]
                        + "_view_{}/surf_normal".format(camera.image_name),
                        [surf_normal[None]],
                        step=iteration,
                    )
                    self.log_image(
                        config["name"]
                        + "_view_{}/rend_alpha".format(camera.image_name),
                        [rend_alpha[None]],
                        step=iteration,
                    )

                    rend_dist = colormap(I.rend_dist.cpu().numpy()[0])
                    self.log_image(
                        config["name"] + "_view_{}/rend_dist".format(camera.image_name),
                        [rend_dist[None]],
                        step=iteration,
                    )

                    if iteration == test_iterations[0]:
                        self.log_image(
                            config["name"]
                            + "_view_{}/ground_truth".format(camera.image_name),
                            [gt_image[None]],
                            step=iteration,
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
            self.log_metrics(metrics=metrics, step=iteration)

        torch.cuda.empty_cache()
