# ShapeFormer: Transformer-based Shape Completion via Sparse Representation

<img src='assets/teaser.jpg'/>

https://user-images.githubusercontent.com/5100481/150949433-40d84ed1-0a8d-4ae4-bd53-8662ebd669fe.mp4

### [Project Page](https://shapeformer.github.io/) | [Paper (ArXiv)](https://arxiv.org/abs/2201.10326) | [Twitter thread](https://twitter.com/yan_xg/status/1539108339422212096)
<!-- | [Pre-trained Models](https://www.dropbox.com/s/we886b1fqf2qyrs/ckpts_ICT.zip?dl=0) :fire: |  -->

**This repository is the official pytorch implementation of our paper, *ShapeFormer: Transformer-based Shape Completion via Sparse Representation*.**

[Xinggaung Yan](http://yanxg.art)<sup>1</sup>,
[Liqiang Lin](https://vcc.tech/people-4)<sup>1</sup>,
[Niloy Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/)<sup>2</sup>,
[Dani Lischinski](https://www.cs.huji.ac.il/~danix/)<sup>3</sup>,
[Danny Cohen-Or](https://danielcohenor.com/)<sup>4</sup>,
[Hui Huang](https://vcc.tech/~huihuang)<sup>1†</sup> <br>
<sup>1</sup>Shenzhen University, <sup>2</sup>University College London, <sup>3</sup>Hebrew University of Jerusalem, <sup>4</sup>Tel Aviv University

## :hourglass_flowing_sand: UPDATES
- [x] Core model code is released, please check core_code/README.md
- [x] **The complete code is released! Please have a try!**
- [ ] Add Google Colab

## Installation
The code is tested in docker enviroment [pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel](https://hub.docker.com/layers/pytorch/pytorch/pytorch/1.6.0-cuda10.1-cudnn7-devel/images/sha256-ccebb46f954b1d32a4700aaeae0e24bd68653f92c6f276a608bf592b660b63d7?context=explore).
The following are instructions for setting up the environment in a Linux system from scratch.
You can also directly pull our provided docker environment: `sudo docker pull qheldiv/shapeformer`
Or build the docker environment by yourself with the setup files in the `Docker` folder.

First, clone this repository with submodule xgutils. [xgutils](https://github.com/QhelDIV/xgutils.git) contains various useful system/numpy/pytorch/3D rendering related functions that will be used by ShapeFormer.

      git clone --recursive https://github.com/QhelDIV/ShapeFormer.git

Then, create a conda environment with the yaml file.

      conda env create -f environment.yaml
      conda activate shapeformer

Next, we need to install torch_scatter through this command

      pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html

## Demo

First, download the pretrained model from this google drive [URL](https://drive.google.com/file/d/1IF7FgmoUnKzGCkAj-iIH8otF3gg_D1ff/view?usp=sharing) and extract the content to experiments/

Then run the following command to test VQDIF. The results are in `experiments/demo_vqdif/results`

      python -m shapeformer.trainer --opts configs/demo/demo_vqdif.yaml --gpu 0 --mode "run"

Run the following command to test ShapeFormer for shape completion. The results are in `experiments/demo_shapeformer/results`

      python -m shapeformer.trainer --opts configs/demo/demo_shapeformer.yaml --gpu 0 --mode "run"

## Dataset

We use the dataset from [IMNet](https://github.com/czq142857/IM-NET#datasets-and-pre-trained-weights), which is obtained from [HSP](https://github.com/chaene/hsp).

The dataset we adopted is a downsampled version (64^3) from these dataset (which is 256 resolution).
Please download our processed dataset from this google drive [URL](https://drive.google.com/file/d/1HUbI45KmXCDJv-YVYxRj-oSPCp0D0xLh/view?usp=sharing).
And then extract the data to `datasets/IMNet2_64/`.

To use the full resolution dataset, please first download the original IMNet and HSP datasets, and run the `make_imnet_dataset` function in `shapeformer/data/imnet_datasets/imnet_datasets.py`

## Usage


First, train VQDIF-16 with 

      python -m shapeformer.trainer --opts configs/vqdif/shapenet_res16.yaml --gpu 0

After VQDIF is trained, train ShapeFormer with

      python -m shapeformer.trainer --opts configs/shapeformer/shapenet_scale.yaml --gpu 0

For testing, you just need to append `--mode test` to the above commands.
And if you only want to run callbacks (such as visualization/generation), set the mode to `run`

There is a visualization callback for both VQDIF and ShapeFormer, who will call the model to obtain 3D meshes and render them to images. The results will be save in `experiments/$exp_name$/results/$callback_name$/`
The callbacks will be automatically called during training and testing, so to get the generation results you just need to test the model.

ALso notice that in the configuration files batch sizes are set to very small so that the model can run on a 12GB memory GPU. You can tune it up if your GPU has a larger memory.

### Multi-GPU
Notice that to use multiple GPUs, just specify the GPU ids. For example `--gpu 0 1 2 4` is to use the 0th, 1st, 2nd, 4th GPU for training. Inside the program their indices will be mapped to 0 1 2 3 for simplicity.


## :notebook_with_decorative_cover: Citation

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@misc{yan2022shapeformer,
      title={ShapeFormer: Transformer-based Shape Completion via Sparse Representation}, 
      author={Xingguang Yan and Liqiang Lin and Niloy J. Mitra and Dani Lischinski and Danny Cohen-Or and Hui Huang},
      year={2022},
      eprint={2201.10326},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 📢: Shout-outs
The architecture of our method is inspired by [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks), [Taming-transformers](https://github.com/CompVis/taming-transformers) and [DCTransformer](https://github.com/benjs/DCTransformer-PyTorch).
Thanks to the authors.

Also, make sure to check this amazing transformer-based image completion project([ICT](https://github.com/raywzy/ICT))!

## :email: Contact

This repo is currently maintained by Xingguang ([@qheldiv](https://github.com/qheldiv)) and is for academic research use only. Discussions and questions are welcome via qheldiv@gmail.com. 
