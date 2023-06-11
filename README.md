# DiffSketch: Plug-and-Play Diffusion Features for Zero-Shot Line-Drawing-to-Sketch Synthesis

> *Stable Diffusion* Implementation, our method is built on PnP-Diffusion

![teaser](assets/results.png)

**To generate Facial Sketch given line drawings, please follow these steps:**

1. [Setup](#setup)
2. [Running synthesis](#running-synthesis)
3. [Gather Results](#gather-results)


## Setup

Our codebase is built on [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
and has shared dependencies and model architecture.

### Creating a Conda Environment

```
conda env create -f environment.yaml
conda activate pnp-diffusion
```

### Downloading StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
mkdir -p models/pnp_ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/pnp_ldm/stable-diffusion-v1/model.ckpt 
```

### Setting Experiment Root Path

The data of all the experiments is stored in a root directory.
The path of this directory is specified in `configs/pnp/setup.yaml`, under the `config.exp_path_root` key.

### Hardware

The code was tested with a single GPU (NVIDIA GeForce RTX 3090) with 24GB of memory.

## Running Synthesis

Note that the data for testing is already provided in the repository (in `data` directory).

### Running on a Single Image

To run the model on a single image, and test with hyperparameters, run the notebook `pnp_sketch.ipynb`. 

Some important parameters to set: `self_attn_output_block_indices`, `out_layers_output_block_indices`, to specify the layers to use for the self-attention and the output layers, respectively; `prompts`, used to specify the prompts to use for the diffusion process; `img_path`, the path to the input image; `exp_config.scale`, used to specify the scale of unconditional guidance (a small value encourage the model to generate a sketch that is close to the input image). More parameters can be found in `configs/pnp_refine_all.yaml`.

### Running on a Dataset

To run the model on a dataset, run the python script `pnp_sketch.py`. 

```bash
python pnp_sketch.py
```

All required hyperparameters are specified in `configs/pnp_refine_all.yaml`.

## Gather Results

To gather the results, run the notebook `gather.ipynb`.

This notebook will create a directory `result` in the root directory, and will gather the results of the experiments in this directory for submission.

## Additional Results

We provide the comparison with [DiffStyle](https://github.com/Junyi42/DiffStyle) and the ablation study as below.

![compare](assets/compare.png)

## Citation
```
@article{pnpDiffusion2022,
  title={Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation},
  author={Tumanyan, Narek and Geyer, Michal and Bagon, Shai and Dekel, Tali},
  journal={arXiv preprint arXiv:2211.12572},
  year={2022}
}
```
