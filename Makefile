
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

MAIN_PART2 = $(PYTHON_INTERPRETER) $(PROJECT_NAME)/part2/main.py
TRAIN_PART2 = $(MAIN_PART2) train
SAMPLE_PART2 = $(MAIN_PART2) sample

DDPM_PARAMS = --model-type ddpm \
	--epochs 10 --batch-size 128

train_ddpm_cuda_from_scratch:
	$(TRAIN_PART2) $(DDPM_PARAMS) --model models/ddpm.pt --device cuda
train_ddpm_cuda_continue:
	$(TRAIN_PART2) $(DDPM_PARAMS) --model models/ddpm.pt --device cuda --continue-train true

sample_ddpm:
	$(SAMPLE_PART2) $(DDPM_PARAMS) --device cuda \
		--samples samples/ddpm/report/ddpm_sample.pdf
sample_ddpm_ep180:
	$(SAMPLE_PART2) $(DDPM_PARAMS) --device cuda \
		--samples samples/ddpm/report/ddpm_sample.pdf --model models/ddpm_ep180.pt

FLOW_MODEL = flow_ep10_rnd_nt20_nh256_hl1.pt
FLOW_PARAMS = --model-type flow \
	--num-transformations 20 --num-hidden 256 --mask-type random \
	--batch-size 32 --epochs 10 \
	--model models/$(FLOW_MODEL)

train_flow:
	$(TRAIN_PART2) $(FLOW_PARAMS)
train_flow_cuda:
	$(TRAIN_PART2) $(FLOW_PARAMS) --device cuda
train_flow_cuda_cont:
	$(TRAIN_PART2) $(FLOW_PARAMS) --device cuda --continue-train true


sample_flow: 
	$(SAMPLE_PART2) $(FLOW_PARAMS)
sample_flow_cuda: 
	$(SAMPLE_PART2) $(FLOW_PARAMS) --device cuda --samples samples/flow/report/flow_sample.pdf

VAE_MODEL = vae_flowP_mvn-dec.pt
VAE_PARAMS = --model-type vae \
	--batch-size 64 --epochs 15 \
	--model models/$(VAE_MODEL)

train_vae_cuda:
	$(TRAIN_PART2) $(VAE_PARAMS) --device cuda
train_vae_cuda_cont:
	$(TRAIN_PART2) $(VAE_PARAMS) --device cuda --continue-train true
sample_vae_cuda:
	$(SAMPLE_PART2) $(VAE_PARAMS) --device cuda --samples samples/vae/report/vae_sample.pdf

batch_sample_ddpm_cuda:
	python $(PROJECT_NAME)/part2/main.py sample_save_batches --model-type ddpm \
	--device cuda --samples samples/ddpm/ddpm_samples.pdf --model models/ddpm_ep180.pt \
	--num-samples 5000 --batch-size 100

batch_sample_flow_cuda:
	python $(PROJECT_NAME)/part2/main.py sample_save_batches $(FLOW_PARAMS) \
	--device cuda --samples samples/flow/flow_samples.pdf --batch-size 100 --num-samples 5000


batch_sample_vae_cuda_flowp:
	python $(PROJECT_NAME)/part2/main.py sample_save_batches $(VAE_PARAMS) --device cuda --samples samples/vae_flow/vae_sample.pdf --num-samples 5000 --batch-size 100

VAE_MODEL_GAUSS = vae_stdG_mvn-dec.pt
VAE_PARAMS_GAUSS = --model-type vae \
	--batch-size 100 --epochs 15 \
	--model models/$(VAE_MODEL_GAUSS)

batch_sample_vae_cuda_gaussp:
	python $(PROJECT_NAME)/part2/main.py sample_save_batches $(VAE_PARAMS_GAUSS) --device cuda --samples samples/vae_gauss/vae_sample.pdf --num-samples 5000


fid_ddpm_cuda:
	python $(PROJECT_NAME)/part2/fid.py --sample-folder samples/ddpm/batch_samples  --device cuda

fid_flow_cuda:
	python $(PROJECT_NAME)/part2/fid.py --sample-folder samples/flow/batch_samples  --device cuda

fid_vae_flowp_cuda:
	python $(PROJECT_NAME)/part2/fid.py --sample-folder samples/vae_flow/batch_samples  --device cuda

fid_vae_gaussp_cuda:
	python $(PROJECT_NAME)/part2/fid.py --sample-folder samples/vae_gauss/batch_samples  --device cuda
