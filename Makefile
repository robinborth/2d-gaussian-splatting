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
	data=debug \
	logger.group=check_loss \
	logger.tags=[check_loss] \
	logger.name=check_loss_wo_gradient \
	task_name=check_loss_wo_gradient \
	model.lambda_gradient=0.0 \
	model.lambda_surface=1.0 \
	model.lambda_empty_space=1.0 \


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
