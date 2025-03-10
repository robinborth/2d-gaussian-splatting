########################################################################
# Ablations
########################################################################

debug:
	python makefile.py & make debug -f Makefile.abl -j 6

all:
	python makefile.py & make all -f Makefile.abl

########################################################################
# Debug 
########################################################################

training:
	python neural_poisson/train.py \
    trainer.max_epochs=50 \
    data.epoch_size=100 \
    data.batch_size=50_000 \
    data.vector_field_mode=k_nearest_neighbors \
    data.k=10 \
    data.max_surface_points=100_000 \
    data.max_close_points=0 \
    data.max_empty_points=0 \
    data.resolution=512 \
    data.sigma=2.0 \
    model/indicator_function=siren \
    model.lambda_gradient=1.0 \
    model.lambda_surface=0.0 \
    model.lambda_empty_space=0.0 \
    model.log_metrics=True \
    model.log_images=True \
    model.log_optimizer=True \
    model.log_mesh=True \
    model.optimizer.lr=1e-04 \
	model.indicator_function.mlp.last_layer_weight_init=False \
    scheduler=exponential_0_99 \


########################################################################
# Mesh Extraction
########################################################################

.PHONY: check_loss check_loss_gradient check_loss_surface check_loss_empty_space check_loss_wo_gradient check_loss_full
check_loss: check_loss_gradient check_loss_surface check_loss_empty_space check_loss_wo_gradient check_loss_full

check_loss_gradient:
	python neural_poisson/train.py \
	data=debug \
	logger.group=check_loss \
	logger.tags=[check_loss] \
	logger.name=check_loss_gradient \
	task_name=check_loss_gradient \
	model.lambda_gradient=1.0 \
	model.lambda_surface=0.0 \
	model.lambda_empty_space=0.0 \

check_loss_surface:
	python neural_poisson/train.py \
	data=debug \
	logger.group=check_loss \
	logger.tags=[check_loss] \
	logger.name=check_loss_surface \
	task_name=check_loss_surface \
	model.lambda_gradient=0.0 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=0.0 \

check_loss_empty_space:
	python neural_poisson/train.py \
	data=debug \
	logger.group=check_loss \
	logger.tags=[check_loss] \
	logger.name=check_loss_empty_space \
	task_name=check_loss_empty_space \
	model.lambda_gradient=0.0 \
	model.lambda_surface=0.0 \
	model.lambda_empty_space=1.0 \

check_loss_wo_gradient:
	python neural_poisson/train.py \
	data.batch_size=10000 \
	logger.group=check_loss \
	logger.tags=[check_loss] \
	logger.name=check_loss_wo_gradient \
	task_name=check_loss_wo_gradient \
	model.lambda_gradient=0.0 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \
	trainer.max_epochs=300 \
	model.optimizer.lr=1e-04 \

check_loss_full:
	python neural_poisson/train.py \
	data=debug \
	logger.group=check_loss \
	logger.tags=[check_loss] \
	logger.name=check_loss_full \
	task_name=check_loss_full \
	model.lambda_gradient=1.0 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \

########################################################################
# Evaluation
########################################################################

.PHONY: train train_full train_wo_gradient train_full_wo_close train_full_wo_close_wo_gradient train_full_wo_close_small_gradient train_full_small_gradient 
train: train_full train_wo_gradient train_full_wo_close train_full_wo_close_wo_gradient train_full_wo_close_small_gradient train_full_small_gradient


train_full:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_full \
	task_name=train_full \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.fov=30.0 \
	data.dist=2.0 \
	data.vector_field_mode=k_nearest_neighbors \
	data.image_size=256 \
	data.resolution=0.0002 \
	data.segments=12 \
	data.max_surface_points=100_000 \
	data.max_close_points=100_000 \
	data.max_empty_points=0 \
	data.sigma=0.001 \
	data.normalize=False \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=1.0 \
	model.lambda_surface=0.0 \
	model.lambda_empty_space=0.0 \
	model.log_metrics=True \
	model.log_images=True \
	model.log_optimizer=True \
	trainer.max_epochs=1000 \
	scheduler=none \


train_wo_gradient:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_wo_gradient \
	task_name=train_wo_gradient \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.fov=30.0 \
	data.dist=2.0 \
	data.vector_field_mode=k_nearest_neighbors \
	data.image_size=256 \
	data.resolution=0.0002 \
	data.segments=12 \
	data.max_surface_points=100_000 \
	data.max_close_points=100_000 \
	data.max_empty_points=100_000 \
	callbacks.model_checkpoint.every_n_epochs=10 \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=0.0 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \
	model.log_metrics=True \
	model.log_images=True \
	model.log_optimizer=True \
	trainer.max_epochs=1000 \
	scheduler=none \

train_full_wo_close:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_full_wo_close \
	task_name=train_full_wo_close \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.fov=30.0 \
	data.dist=2.0 \
	data.vector_field_mode=k_nearest_neighbors \
	data.image_size=256 \
	data.resolution=0.0002 \
	data.segments=12 \
	data.max_surface_points=100_000 \
	data.max_close_points=0 \
	data.max_empty_points=100_000 \
	callbacks.model_checkpoint.every_n_epochs=10 \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=1e-03 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \
	model.log_metrics=True \
	model.log_images=True \
	model.log_optimizer=True \
	trainer.max_epochs=1000 \
	scheduler=none \

train_full_wo_close_wo_gradient:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_full_wo_close_wo_gradient \
	task_name=train_full_wo_close_wo_gradient \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.fov=30.0 \
	data.dist=2.0 \
	data.vector_field_mode=k_nearest_neighbors \
	data.image_size=256 \
	data.resolution=0.0002 \
	data.segments=12 \
	data.max_surface_points=100_000 \
	data.max_close_points=0 \
	data.max_empty_points=100_000 \
	callbacks.model_checkpoint.every_n_epochs=10 \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=0.0 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \
	model.log_metrics=True \
	model.log_images=True \
	model.log_optimizer=True \
	trainer.max_epochs=2000 \
	scheduler=none \

train_full_wo_close_small_gradient:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_full_wo_close_small_gradient \
	task_name=train_full_wo_close_small_gradient \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.fov=30.0 \
	data.dist=2.0 \
	data.vector_field_mode=k_nearest_neighbors \
	data.image_size=256 \
	data.resolution=0.0002 \
	data.segments=12 \
	data.max_surface_points=100_000 \
	data.max_close_points=0 \
	data.max_empty_points=100_000 \
	callbacks.model_checkpoint.every_n_epochs=10 \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=1e-06 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \
	model.log_metrics=True \
	model.log_images=True \
	model.log_optimizer=True \
	trainer.max_epochs=1000 \
	scheduler=none \

train_full_small_gradient:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_full_small_gradient \
	task_name=train_full_small_gradient \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.fov=30.0 \
	data.dist=2.0 \
	data.vector_field_mode=k_nearest_neighbors \
	data.image_size=256 \
	data.resolution=0.0002 \
	data.segments=12 \
	data.max_surface_points=100_000 \
	data.max_close_points=100_000 \
	data.max_empty_points=100_000 \
	callbacks.model_checkpoint.every_n_epochs=10 \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=1e-06 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \
	model.log_metrics=True \
	model.log_images=True \
	model.log_optimizer=True \
	trainer.max_epochs=1000 \
	scheduler=none \


train_only_gradient:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_only_gradient \
	task_name=train_only_gradient \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.fov=30.0 \
	data.dist=2.0 \
	data.vector_field_mode=k_nearest_neighbors \
	data.image_size=256 \
	data.resolution=0.0002 \
	data.segments=12 \
	data.max_surface_points=100_000 \
	data.max_close_points=100_000 \
	data.max_empty_points=100_000 \
	callbacks.model_checkpoint.every_n_epochs=10 \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=1.0 \
	model.lambda_surface=0.0 \
	model.lambda_empty_space=0.0 \
	model.log_metrics=True \
	model.log_images=True \
	model.log_optimizer=True \
	trainer.max_epochs=1000 \
	scheduler=none \

