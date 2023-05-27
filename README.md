# DiffSketch: Leverage Diffusion Prior to Zero-Shot Line-Drawing-to-Sketch Synthesis

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



## Running Synthesis


## Gather Results


## Citation
```
@article{pnpDiffusion2022,
  title={Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation},
  author={Tumanyan, Narek and Geyer, Michal and Bagon, Shai and Dekel, Tali},
  journal={arXiv preprint arXiv:2211.12572},
  year={2022}
}
```
