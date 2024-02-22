
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
	PYTHON_INTERPRETER -m venv venv

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
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py train --device cuda --batch-size 128 --epochs 10 --model models/ddpm.pt
train_ddpm_cuda_continue:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py train --device cuda --batch-size 128 --epochs 10 --model models/ddpm.pt --continue-train true

sample_ddpm:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py sample --device mps --samples samples/ddpm_samples.png --model models/ddpm.pt
sample_ddpm_ep180:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py sample --device mps --samples samples/ddpm_samples.png --model models/ddpm_ep180.pt