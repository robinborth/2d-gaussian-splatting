from typing import Any

import hydra
from omegaconf import DictConfig


class MakefileGenerator:
    def __init__(self, suffix: str = "abl", debug_template: str | None = None):
        self.suffix = suffix
        self.groups: list[Any] = []
        self.debug_template = debug_template

    @property
    def debug(self):
        return self.debug_template is not None

    def _generate_banner(self, group_names: list[str]):
        b = ""
        b += "##########################################################################\n"
        b += f"# make all -f Makefile.{self.suffix}\n"
        for i, group in enumerate(group_names):
            b += f"# GROUP{i}: make {group} -f Makefile.{self.suffix}\n"
        b += "##########################################################################\n"
        return b

    def _generate_group_banner(self, group_name: str = "GROUP"):
        b = ""
        b += "##########################################################################\n"
        b += f"# {group_name}\n"
        b += "##########################################################################\n"
        b += "\n"
        return b

    def _generate_make_command(self, template_generator, value, group_name, prefix):
        task_name = f"{group_name}__{prefix}"
        make_command = f"{task_name}:"
        template = template_generator.format(
            task_name=task_name,
            value=value,
            group_name=group_name,
        )
        template = make_command + template
        return task_name, template

    def convert_float_to_scientific(self, values):
        return [f"{v:.1E}".replace(".", "-") for v in values]

    def _add(
        self,
        template_generator,
        values,
        prefixs,
        group_name: str = "GROUP",
    ):
        templates = []
        task_names = []
        for value, prefix in zip(values, prefixs):
            task_name, template = self._generate_make_command(
                template_generator=template_generator,
                value=value,
                group_name=group_name,
                prefix=prefix,
            )
            t = "\n\t".join(
                [li.strip() for li in template.split("\n") if li and li.strip()]
            )
            task_names.append(task_name)
            templates.append(t)

        temp = self._generate_group_banner(group_name)
        tasks = " ".join(task_names)
        temp += f"\n{group_name}: {tasks}\n\n"
        temp += "\n\n".join(templates)
        temp += "\n"

        self.groups.append([temp, group_name, tasks])

    def add(
        self,
        template_generator,
        values,
        prefixs,
        group_name: str = "GROUP",
    ):
        self._add(template_generator, values, prefixs, group_name)
        if self.debug:
            group_name = f"DEBUG_{group_name}"
            template_generator += self.debug_template
            self._add(template_generator, values, prefixs, group_name)

    def build(self) -> None:
        templates, group_names, task_names = [], [], []
        debug_templates, debug_task_names = [], []
        for group in self.groups:
            temp, names, task = group
            group_names.append(names)
            if names.startswith("DEBUG"):
                debug_templates.append(temp)
                debug_task_names.append(task)
            else:
                templates.append(temp)
                task_names.append(task)

        g = self._generate_banner(group_names)

        g += "\n"
        tasks = " ".join(task_names)
        g += f".PHONY: all {tasks}\n"
        g += f"all: {tasks}\n\n"

        if self.debug:
            tasks = " ".join(debug_task_names)
            g += f".PHONY: debug {tasks}\n"
            g += f"debug: {tasks}\n\n"

        g += "\n\n".join(templates)
        g += "\n\n"

        if self.debug:
            g += "\n\n".join(debug_templates)
            g += "\n\n"

        with open(f"Makefile.{self.suffix}", "w") as f:
            f.write(g)


@hydra.main(version_base=None, config_path="./conf", config_name="makefile")
def main(cfg: DictConfig):
    makefile_generator = MakefileGenerator(suffix=cfg.suffix)
    makefile_generator.debug_template = """
    trainer.max_epochs=10 \\
    data.epoch_size=1 \\
    data.dataset.vector_field_mode=nearest_neighbor \\
    data.dataset.image_size=128 \\
    data.dataset.resolution=0.05 \\
    data.dataset.segments=4 \\
    callbacks.model_checkpoint.every_n_epochs=5 \\
    """

    values = ["relu", "gelu", "sin", "cos"]
    prefixs = values
    group_name = "sigmoid_mlp"
    template_generator = """
	data.epoch_size=100 \\
	data.batch_size=50_000 \\
	data.dataset.fov=30.0 \\
	data.dataset.dist=2.0 \\
	data.dataset.image_size=256 \\
	data.dataset.segments=12 \\
	data.dataset.k=10 \\
	data.dataset.vector_field_mode=k_nearest_neighbors \\
	data.dataset.max_surface_points=100_000 \\
	data.dataset.max_close_points=0 \\
	data.dataset.max_empty_points=0 \\
	data.dataset.resolution=0.001 \\
	data.dataset.sigma=0.001 \\
	model.optimizer.lr=1e-04 \\
	model.lambda_gradient=1.0 \\
	model.lambda_surface=0.0 \\
	model.lambda_empty_space=0.0 \\
	model.log_metrics=True \\
	model.log_images=True \\
	model.log_optimizer=True \\
	model.log_mesh=False \\
	model.activation=sigmoid \\
	model.encoder.activation={value} \\
	trainer.max_epochs=50 \\
	trainer.detect_anomaly=False \\
	scheduler=none \\
    """
    makefile_generator.add(template_generator, values, prefixs, group_name)

    values = ["relu", "gelu", "sin", "cos"]
    prefixs = values
    group_name = "sin_mlp"
    template_generator = """
	data.epoch_size=100 \\
	data.batch_size=50_000 \\
	data.dataset.fov=30.0 \\
	data.dataset.dist=2.0 \\
	data.dataset.image_size=256 \\
	data.dataset.segments=12 \\
	data.dataset.k=10 \\
	data.dataset.vector_field_mode=k_nearest_neighbors \\
	data.dataset.max_surface_points=100_000 \\
	data.dataset.max_close_points=0 \\
	data.dataset.max_empty_points=0 \\
	data.dataset.resolution=0.001 \\
	data.dataset.sigma=0.001 \\
	model.optimizer.lr=1e-04 \\
	model.lambda_gradient=1.0 \\
	model.lambda_surface=0.0 \\
	model.lambda_empty_space=0.0 \\
	model.log_metrics=True \\
	model.log_images=True \\
	model.log_optimizer=True \\
	model.log_mesh=False \\
	model.activation=sigmoid \\
	model.encoder.activation={value} \\
	trainer.max_epochs=50 \\
	trainer.detect_anomaly=False \\
	scheduler=none \\
    """
    makefile_generator.add(template_generator, values, prefixs, group_name)

    makefile_generator.build()


if __name__ == "__main__":
    main()
