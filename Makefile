# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions
conda-update:
	conda env update --prune -f environment.yml

# Compile and install exact pip packages
pip-tools:
	python -m pip install pip-tools
	# python -m pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
	pip-compile --find-links=https://download.pytorch.org/whl/torch_stable.html requirements/prod.in && pip-compile requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt

# Example training command
train:
#	python training/run_experiment.py --max_epochs=10 --gpus=-1 --accelerator=ddp --num_workers=20 --data_class=MNIST --model_class=MLP
# python training/run_experiment.py --max_epochs=10 --gpus=1 --accelerator=ddp --num_workers=20 --data_class=CASIA --model_class=CNN
	python training/run_experiment.py --max_epochs=3 --gpus='1,' --data_class=CASIA --model_class=CNN

run:
	rm /storage/tmp/*
	python app.py
# Lint
lint:
	tasks/lint.sh
