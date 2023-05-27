import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
try:
    from torch import autocast
except:
    from torch.cuda.amp import autocast
from contextlib import nullcontext
import json
from torchvision import transforms
import logging
from pnp_utils import check_safety

from pnp_ldm.util import instantiate_from_config
from pnp_ldm.models.diffusion.ddim import DDIMSampler
import collections

memory_storage = collections.defaultdict(dict)

def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="/hdd/junyi/plug-and-play/configs/refine",
        help="path to the feature extraction config file"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--save_all_features",
        action="store_true",
        help="if set to true, saves all feature maps, otherwise only saves those necessary for PnP",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--check-safety",
        action='store_true',
    )

    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/pnp/setup.yaml")
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)

    config_dir_path=opt.config
    config_paths=[]
    
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(config_dir_path):
        for file in files:
            # 如果文件名以".config"结尾，则将路径添加到列表中
            if file.endswith(".yaml"):
                path = os.path.join(root, file)
                config_paths.append(path)


    for config_path in config_paths:
        exp_config = OmegaConf.load(config_path)
        
        exp_path_root = setup_config.config.exp_path_root

        if exp_config.config.init_img != '':
            exp_config.config.seed = -1
            exp_config.config.prompt = ""
            exp_config.config.scale = 1.0
            
        seed = exp_config.config.seed 
        seed_everything(seed)

        save_feature_timesteps = exp_config.config.ddim_steps if exp_config.config.init_img == '' else exp_config.config.save_feature_timesteps

        outpath = f"{exp_path_root}/{exp_config.config.experiment_name}"

        callback_timesteps_to_save = [save_feature_timesteps]
        if os.path.exists(outpath):
            logging.warning("Experiment directory already exists, previously saved content will be overriden")
            if exp_config.config.init_img != '':
                with open(os.path.join(outpath, "args.json"), "r") as f:
                    args = json.load(f)
                callback_timesteps_to_save = args["save_feature_timesteps"] + callback_timesteps_to_save

        predicted_samples_path = os.path.join(outpath, "predicted_samples")
        feature_maps_path = os.path.join(outpath, "feature_maps")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(predicted_samples_path, exist_ok=True)
        os.makedirs(feature_maps_path, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)

        # save parse_args in experiment dir
        with open(os.path.join(outpath, "args.json"), "w") as f:
            args_to_save = OmegaConf.to_container(exp_config.config)
            args_to_save["save_feature_timesteps"] = callback_timesteps_to_save
            json.dump(args_to_save, f)

        def save_sampled_img(x, i, save_path):
            x_samples_ddim = model.decode_first_stage(x)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
            x_sample = x_image_torch[0]
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(save_path, f"{i}.png"))

        def ddim_sampler_callback(pred_x0, x_t, i):
            save_feature_maps_callback(i)
            save_sampled_img(pred_x0, i, predicted_samples_path)

        def save_feature_maps(blocks, i, feature_type="input_block"):
            block_idx = 0
            for block in tqdm(blocks, desc="Saving input blocks feature maps"):
                if not opt.save_all_features and block_idx < 4:
                    block_idx += 1
                    continue
                if "ResBlock" in str(type(block[0])):
                    if opt.save_all_features or block_idx == 4:
                        save_feature_map(block[0].in_layers_features, f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                        save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
                if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                    save_feature_map(block[1].transformer_blocks[0].attn1.k, f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                    save_feature_map(block[1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
                block_idx += 1

        def save_feature_maps_callback(i):
            if opt.save_all_features:
                save_feature_maps(unet_model.input_blocks, i, "input_block")
            save_feature_maps(unet_model.output_blocks , i, "output_block")

        def save_feature_map(feature_map, filename):
            memory_storage[filename] = feature_map

        assert exp_config.config.prompt is not None
        prompts = [exp_config.config.prompt]

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    uc = model.get_learned_conditioning([""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                    z_enc = None
                    if exp_config.config.init_img != '':
                        assert os.path.isfile(exp_config.config.init_img)
                        init_image = load_img(exp_config.config.init_img).to(device)
                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
                        ddim_inversion_steps = 999
                        z_enc, _ = sampler.encode_ddim(init_latent, num_steps=ddim_inversion_steps, conditioning=c,unconditional_conditioning=uc,unconditional_guidance_scale=exp_config.config.scale)
                    else:
                        z_enc = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                    torch.save(z_enc, f"{outpath}/z_enc.pt")
                    samples_ddim, _ = sampler.sample(S=exp_config.config.ddim_steps,
                                    conditioning=c,
                                    batch_size=1,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=exp_config.config.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta,
                                    x_T=z_enc,
                                    img_callback=ddim_sampler_callback,
                                    callback_ddim_timesteps=save_feature_timesteps,
                                    outpath=outpath)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    if opt.check_safety:
                        x_samples_ddim = check_safety(x_samples_ddim)
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    sample_idx = 0
                    for x_sample in x_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(sample_path, f"{sample_idx}.png"))
                        sample_idx += 1

        print(f"Sampled images and extracted features saved in: {outpath}")
        exp_path_root_config = OmegaConf.load("./configs/pnp/setup.yaml")
        exp_path_root = exp_path_root_config.config.exp_path_root
        
        # read seed from args.json of source experiment
        with open(os.path.join(exp_path_root, exp_config.config.experiment_name, "args.json"), "r") as f:
            args = json.load(f)
            seed = args["seed"]
            source_prompt = args["prompt"]
        negative_prompt = source_prompt if exp_config.negative_prompt is None else exp_config.negative_prompt

        seed_everything(seed)
        possible_ddim_steps = args["save_feature_timesteps"]
        assert exp_config.num_ddim_sampling_steps in possible_ddim_steps or exp_config.num_ddim_sampling_steps is None, f"possible sampling steps for this experiment are: {possible_ddim_steps}; for {exp_config.num_ddim_sampling_steps} steps, run 'run_features_extraction.py' with save_feature_timesteps = {exp_config.num_ddim_sampling_steps}"
        ddim_steps = exp_config.num_ddim_sampling_steps if exp_config.num_ddim_sampling_steps is not None else possible_ddim_steps[-1]
        
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 

        seed = torch.initial_seed()
        opt.seed = seed

        translation_folders = [p.replace(' ', '_') for p in exp_config.prompts]
        outpaths = [os.path.join(f"{exp_path_root}/{exp_config.config.experiment_name}/translations", f"{exp_config.scale}_{translation_folder}") for translation_folder in translation_folders]
        out_label = f"INJECTION_T_{exp_config.feature_injection_threshold}_STEPS_{ddim_steps}"
        out_label += f"_NP-ALPHA_{exp_config.negative_prompt_alpha}_SCHEDULE_{exp_config.negative_prompt_schedule}_NP_{negative_prompt.replace(' ', '_')}"

        predicted_samples_paths = [os.path.join(outpath, f"predicted_samples_{out_label}") for outpath in outpaths]
        for i in range(len(outpaths)):
            os.makedirs(outpaths[i], exist_ok=True)
            os.makedirs(predicted_samples_paths[i], exist_ok=True)
            # save args in experiment dir
            with open(os.path.join(outpaths[i], "args.json"), "w") as f:
                json.dump(OmegaConf.to_container(exp_config), f)

        def save_sampled_img(x, i, save_paths):
            for im in range(x.shape[0]):
                x_samples_ddim = model.decode_first_stage(x[im].unsqueeze(0))
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                x_sample = x_image_torch[0]

                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(os.path.join(save_paths[im], f"{i}.png"))

        def ddim_sampler_callback(pred_x0, xt, i):
            save_sampled_img(pred_x0, i, predicted_samples_paths)

        def load_feature_map(filename):
            return memory_storage[filename]
        
        def load_target_features():
            self_attn_output_block_indices = [4, 5, 6, 7, 8, 9, 10, 11]
            out_layers_output_block_indices = [4]
            output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(self_attn_output_block_indices)
            feature_injection_thresholds = [exp_config.feature_injection_threshold]
            target_features = []

            time_range = np.flip(sampler.ddim_timesteps)
            total_steps = sampler.ddim_timesteps.shape[0]

            iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)

            for i, t in enumerate(iterator):
                current_features = {}
                for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                    if i <= int(output_block_self_attn_map_injection_threshold):
                        output_q = load_feature_map(f"output_block_{output_block_idx}_self_attn_q_time_{t}")
                        output_k = load_feature_map(f"output_block_{output_block_idx}_self_attn_k_time_{t}")
                        current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                        current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

                for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
                    if i <= int(feature_injection_threshold):
                        output = load_feature_map(f"output_block_{output_block_idx}_out_layers_features_time_{t}")
                        current_features[f'output_block_{output_block_idx}_out_layers'] = output

                target_features.append(current_features)

            return target_features # for one timestep, attn: (16, patch**2, dim); out_layers: (2, 1280, 16, 16)

        batch_size = len(exp_config.prompts)
        prompts = exp_config.prompts
        assert prompts is not None

        start_code_path = f"{exp_path_root}/{exp_config.config.experiment_name}/z_enc.pt"
        start_code = torch.load(start_code_path).cuda() if os.path.exists(start_code_path) else None
        if start_code is not None:
            start_code = start_code.repeat(batch_size, 1, 1, 1)

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        injected_features = load_target_features()
        unconditional_prompt = ""
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    uc = None
                    nc = None
                    if exp_config.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [unconditional_prompt])
                        nc = model.get_learned_conditioning(batch_size * [negative_prompt])
                    if not isinstance(prompts, list):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    negative_conditioning=nc,
                                                    batch_size=len(prompts),
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=exp_config.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code,
                                                    img_callback=ddim_sampler_callback,
                                                    injected_features=injected_features,
                                                    negative_prompt_alpha=exp_config.negative_prompt_alpha,
                                                    negative_prompt_schedule=exp_config.negative_prompt_schedule,
                                                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    if opt.check_safety:
                        x_samples_ddim = check_safety(x_samples_ddim)
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    sample_idx = 0
                    for k, x_sample in enumerate(x_image_torch):
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(outpaths[k], f"{out_label}_sample_{sample_idx}.png"))
                        sample_idx += 1

        print(f"PnP results saved in: {'; '.join(outpaths)}")


if __name__ == "__main__":
    main()
