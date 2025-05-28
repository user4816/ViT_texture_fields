## What's New
Based on the original FM-ONet: Texture Fields implementation, we introduce enhancements and adjustments to improve model performance. Specifically, users can modify encoder and decoder configurations in the provided YAML files to achieve optimized texture generation results. Our modifications allow clearer, higher-quality outputs for diverse single-view reconstruction tasks.

##Installation
conda env create -f enh_textfield.yaml
conda activate Enh_textfield

##Inference (Generation)
# Generate results using a pretrained model.
# Checkpoint path is defined in 'configs/singleview/texfields/car_demo.yaml'.

python generate.py configs/singleview/texfields/car_demo.yaml


## Training
# Adjust '--nproc_per_node' based on your available CUDA devices.
# Before training, configure encoder and decoder settings in 'configs/singleview/texfields/car.yaml'.

python -m torch.distributed.launch --nproc_per_node=8 train.py configs/singleview/texfields/car.yaml
