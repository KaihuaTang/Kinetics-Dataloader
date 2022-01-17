# A self-contained pytorch Kinetics Dataloader
This project provides a compact self-contained kinetics dataloader based on Pytorch. The codes are modified from [SlowFast](https://github.com/facebookresearch/SlowFast).

# Environments
Run the following commands to install the exvironments
```
conda create -n YOUR_ENV_NAME pip python=3.8
source activate YOUR_ENV_NAME
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
conda install pillow
conda install matplotlib
pip install pyyaml
pip install av
```

# Prepare Datasets
Follow the [Dataset Preprocessing](https://github.com/KaihuaTang/Kinetics-Data-Preprocessing) to prepare the Kinetics-400/Kinetics-600 datasets.

# Demo
Run the [Demo](https://github.com/KaihuaTang/Kinetics-Dataloader/blob/main/demo.ipynb) to visualize the sampled video frames. This project keep the most compact version of dataloader by removing all the frame-level data augmentation.