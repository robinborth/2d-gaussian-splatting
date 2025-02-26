from typing import Any

import hydra
from omegaconf import DictConfig


class MakefileGenerator:
    def __init__(
        self,
        suffix: str = "abl",
        debug_template: str | None = None,
        default_template: str | None = None,
        logging_template: str | None = None,
        script: str | None = None,
    ):
        self.suffix = suffix
        self.groups: list[Any] = []

        # check if you want to create debug template
        self.debug_template = debug_template

        self.default_template = ""
        if default_template is not None:
            self.default_template = default_template

        self.logging_template = """
        logger.group={group_name} \\
        logger.name={task_name} \\
        logger.tags=[{group_name},{task_name}] \\
        task_name={task_name} \\
        """
        if logging_template is not None:
            self.logging_template = logging_template

        self.script = """
        python train.py \\
        """
        if script is not None:
            self.script = script

    @property
    def debug(self):
        return self.debug_template is not None

    def _generate_banner(self, group_names: list[str]):
        b = ""
        b += "##########################################################################\n"
        b += f"# make all -f Makefile.{self.suffix}\n"
        for i, group in enumerate(group_names):
            if not group.startswith("DEBUG"):
                b += f"# GROUP{i}: make {group} -f Makefile.{self.suffix}\n"
        for i, group in enumerate(group_names):
            if group.startswith("DEBUG"):
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

    def _generate_make_command(
        self,
        group: str,
        template: str,
        prefix: str,
        values: dict[str, str],
    ):
        task_name = f"{group}__{prefix}"
        setting_template = template.format(**values)
        logging_template = self.logging_template.format(
            task_name=task_name,
            group_name=group,
        )
        template = (
            f"{task_name}:"
            + self.script
            + logging_template
            + self.default_template
            + setting_template
        )
        return task_name, template

    def convert_float_to_scientific(self, values):
        return [f"{v:.1E}".replace(".", "-") for v in values]

    def _add(
        self,
        group: str,
        template: str,
        prefixs: list[str],
        values: dict[str, list[str]],
    ):
        templates = []
        task_names = []
        for idx, prefix in enumerate(prefixs):
            task_name, t = self._generate_make_command(
                group=group,
                template=template,
                prefix=prefix,
                values={k: v[idx] for k, v in values.items()},
            )
            t = "\n\t".join([li.strip() for li in t.split("\n") if li and li.strip()])
            task_names.append(task_name)
            templates.append(t)

        temp = self._generate_group_banner(group)
        tasks = " ".join(task_names)
        temp += f"\n{group}: {tasks}\n\n"
        temp += "\n\n".join(templates)
        temp += "\n"

        self.groups.append([temp, group, tasks])

    def add(
        self,
        group: str,
        template: str,
        prefixs: list[str],
        **values: list[str],
    ):
        self._add(group, template, prefixs, values)
        if self.debug:
            group = f"DEBUG_{group}"
            template += self.debug_template
            self._add(group, template, prefixs, values)

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
    makefile_generator.script = """
    python neural_poisson/train.py \\
    """
    makefile_generator.default_template = """
    trainer.max_epochs=200 \\
    data.epoch_size=100 \\
    data.batch_size=50_000 \\
    data.dataset.vector_field_mode=nearest_neighbor \\
    data.dataset.max_surface_points=100_000 \\
    data.dataset.max_close_points=0 \\
    data.dataset.max_empty_points=0 \\
    data.dataset.resolution=0.001 \\
    data.dataset.sigma=0.001 \\
    model.lambda_gradient=1.0 \\
    model.lambda_surface=0.0 \\
    model.lambda_empty_space=0.0 \\
    model.log_metrics=True \\
    model.log_images=True \\
    model.log_optimizer=True \\
    model.log_mesh=True \\
    model.optimizer.lr=5e-05 \\
    model.activation=sigmoid \\
    encoder/activation=siren \\
    scheduler=none \\
    """
    makefile_generator.debug_template = """
    trainer.max_epochs=10 \\
    data.epoch_size=1 \\
    data.batch_size=1_000 \\
    data.dataset.segments=4 \\
    data.dataset.image_size=128 \\
    data.dataset.resolution=0.05 \\
    data.dataset.log_camera_idxs=[0] \\
    model.log_mesh=False \\
    callbacks.model_checkpoint.every_n_epochs=10 \\
    """
    value: Any = None

    group = "surface_objective"
    value = [1e2, 1e1, 1e0, 1e-01, 1e-02, 1e-03]
    prefix = makefile_generator.convert_float_to_scientific(value)
    template = "model.lambda_surface={value}"
    makefile_generator.add(group, template, prefix, value=value)

    group = "empty_objective"
    value = [1e2, 1e1, 1e0, 1e-01, 1e-02, 1e-03]
    prefix = makefile_generator.convert_float_to_scientific(value)
    template = """
    model.lambda_empty_space={value} \\
    data.dataset.max_empty_points=100_000 \\
    """
    makefile_generator.add(group, template, prefix, value=value)

    group = "empty_surface_objective"
    value = [1e2, 1e1, 1e0, 1e-01, 1e-02, 1e-03]
    prefix = makefile_generator.convert_float_to_scientific(value)
    template = """
    model.lambda_surface={value} \\
    model.lambda_empty_space={value} \\
    data.dataset.max_empty_points=100_000 \\
    """
    makefile_generator.add(group, template, prefix, value=value)

    group = "only_empty_surface_objective_learning_rate"
    value = [
        1e-04,
        9e-05,
        7e-05,
        5e-05,
        3e-05,
        1e-05,
    ]
    prefix = makefile_generator.convert_float_to_scientific(value)
    template = """
    model.lambda_gradient=0.0 \\
    model.lambda_surface={value} \\
    model.lambda_empty_space={value} \\
    data.dataset.max_empty_points=100_000 \\
    """
    makefile_generator.add(group, template, prefix, value=value)

    group = "activation"
    value = ["tanh", "sinus", "cosine", "siren", "gelu", "relu"]
    prefix = value
    template = "encoder/activation={value}"
    makefile_generator.add(group, template, prefix, value=value)

    value = [
        1e-04,
        9e-05,
        7e-05,
        5e-05,
        3e-05,
        1e-05,
    ]
    prefix = makefile_generator.convert_float_to_scientific(value)
    for scheduler in ["none", "linear", "exponential"]:
        group = f"siren_learning_rate_{scheduler}"
        template = "model.optimizer.lr={value} \\"
        template += f"\nscheduler={scheduler} \\"
        makefile_generator.add(group, template, prefix, value=value)

    makefile_generator.build()


if __name__ == "__main__":
    main()
