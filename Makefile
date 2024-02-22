
.PHONY: requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_venv:
	$(PYTHON_INTERPRETER) -m venv venv

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

train_ddpm_cuda_from_scratch:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py train --model-type ddpm --device cuda \
		--batch-size 128 --epochs 10 --model models/ddpm.pt
train_ddpm_cuda_continue:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py train --model-type ddpm --device cuda \
		--batch-size 128 --epochs 10 --model models/ddpm.pt \
		--continue-train true

sample_ddpm:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py sample --model-type ddpm --device mps \
		--samples samples/ddpm/ddpm_samples.pdf --model models/ddpm.pt
sample_ddpm_ep180:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py sample --model-type ddpm --device mps \
		--samples samples/ddpm/ddpm_samples.pdf --model models/ddpm_ep180.pt
	
TRAIN_FLOW_CMD = python $(PROJECT_NAME)/part2/main.py train --model-type flow \
		--num-transformations 30 --num-hidden 256 --mask-type cb \
		--batch-size 32 --epochs 5 --model models/flow.pt
train_flow:
	$(TRAIN_FLOW_CMD)
train_flow_cuda:
	$(TRAIN_FLOW_CMD) --device cuda

sample_flow: 
	python $(PROJECT_NAME)/part2/main.py sample --model-type flow \
		--device mps --samples samples/flow/flow_samples.pdf --mask-type cb \
		--model models/flow.pt
sample_flow_cuda: 
	python $(PROJECT_NAME)/part2/main.py sample --model-type flow \
		--device cuda --samples samples/flow/flow_samples.pdf  \
		--num-transformations 30 --num-hidden 256 --mask-type random \
		--model models/flow.pt
