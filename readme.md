### FM-ONet: Texture Fields

conda env create -f Enh_textfield.yaml
conda activate Enh_textfield

### Train
## Adjust --nproc_per_node based on visible CUDA devices.
## Note: Before training, modify encoder and decoder settings in configs/singleview/texfields/car.ymal
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/singleview/texfields/car.yaml


### Generate (Inference)
## Note: This step uses a pretrained model checkpoint specified in configs/demo.yaml.
python generate.py configs/singleview/texfields/car_demo.yaml