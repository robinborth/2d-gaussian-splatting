########################################################################
# Optimization 
########################################################################


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
	data.dataset.batch_size=10000 \
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

train_full:
	python neural_poisson/train.py \
	logger.group=train \
	logger.tags=[train] \
	logger.name=train_full \
	task_name=train_full \
	data.epoch_size=100 \
	data.batch_size=50_000 \
	data.dataset.fov=30.0 \
	data.dataset.dist=2.0 \
	data.dataset.vector_field_mode=k_nearest_neighbors \
	data.dataset.image_size=256 \
	data.dataset.resolution=0.0002 \
	data.dataset.segments=12 \
	data.dataset.max_surface_points=100_000 \
	data.dataset.max_close_points=100_000 \
	data.dataset.max_empty_points=100_000 \
	callbacks.model_checkpoint.every_n_epochs=10 \
	model.optimizer.lr=1e-04 \
	model.lambda_gradient=1.0 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \
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
	data.dataset.fov=30.0 \
	data.dataset.dist=2.0 \
	data.dataset.vector_field_mode=k_nearest_neighbors \
	data.dataset.image_size=256 \
	data.dataset.resolution=0.0002 \
	data.dataset.segments=12 \
	data.dataset.max_surface_points=100_000 \
	data.dataset.max_close_points=100_000 \
	data.dataset.max_empty_points=100_000 \
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