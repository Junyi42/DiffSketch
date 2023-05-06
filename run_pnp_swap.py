import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange, tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
from pnp_utils import check_safety

from pnp_ldm.util import instantiate_from_config
from pnp_ldm.models.diffusion.ddim import DDIMSampler
from run_features_extraction import load_model_from_config
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = '/hdd/junyi/plug-and-play/configs/pnp/pnp-generated_swap.yaml')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--model_config', type=str, default='configs/stable-diffusion/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument("--check-safety", action='store_true')
    opt = parser.parse_args()
    exp_config = OmegaConf.load(opt.config)

    exp_path_root_config = OmegaConf.load("./configs/pnp/setup.yaml")
    exp_path_root = exp_path_root_config.config.exp_path_root
    
    # read seed from args.json of source experiment
    with open(os.path.join(exp_path_root, exp_config.source_experiment_name, "args.json"), "r") as f:
        args = json.load(f)
        seed = args["seed"]
        # source_prompt = args["prompt"]
    with open(os.path.join(exp_path_root, exp_config.target_experimen_name, "args.json"), "r") as f:
        args2 = json.load(f)
        target_prompt = args2["prompt"]
    negative_prompt = target_prompt if exp_config.negative_prompt is None else exp_config.negative_prompt

    seed_everything(seed)
    possible_ddim_steps = args["save_feature_timesteps"]
    assert exp_config.num_ddim_sampling_steps in possible_ddim_steps or exp_config.num_ddim_sampling_steps is None, f"possible sampling steps for this experiment are: {possible_ddim_steps}; for {exp_config.num_ddim_sampling_steps} steps, run 'run_features_extraction.py' with save_feature_timesteps = {exp_config.num_ddim_sampling_steps}"
    ddim_steps = exp_config.num_ddim_sampling_steps if exp_config.num_ddim_sampling_steps is not None else possible_ddim_steps[-1]
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 

    seed = torch.initial_seed()
    opt.seed = seed

    translation_folders = [p.replace(' ', '_') for p in exp_config.prompts]
    outpaths = [os.path.join(f"{exp_path_root}/{exp_config.source_experiment_name}/translations", f"{exp_config.scale}_{translation_folder}") for translation_folder in translation_folders]
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

    def load_target_features():
        self_attn_output_block_indices = [4,5,6,7,8,9,10,11]
        out_layers_output_block_indices = [4]
        output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(self_attn_output_block_indices)
        feature_injection_thresholds = [exp_config.feature_injection_threshold]
        target_features = []

        source_experiment_out_layers_path = os.path.join(exp_path_root, exp_config.source_experiment_name, "feature_maps")
        source_experiment_qkv_path = os.path.join(exp_path_root, exp_config.source_experiment_name, "feature_maps")

        target_experiment_out_layers_path = os.path.join(exp_path_root, exp_config.target_experiment_name, "feature_maps")
        target_experiment_qkv_path = os.path.join(exp_path_root, exp_config.target_experiment_name, "feature_maps")

        time_range = np.flip(sampler.ddim_timesteps)
        total_steps = sampler.ddim_timesteps.shape[0]

        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)
        
        src_feature=torch.load(os.path.join(exp_path_root, exp_config.source_experiment_name,"fused_feature.pt")) # (1,1,3600,768*2)
        tgt_feature=torch.load(os.path.join(exp_path_root, exp_config.target_experiment_name,"fused_feature.pt")) # (1,1,3600,768*2)
        src_mask=torch.load(os.path.join(exp_path_root, exp_config.source_experiment_name,"mask.pt")) # mask1 shape (840,840)
        tgt_mask=torch.load(os.path.join(exp_path_root, exp_config.target_experiment_name,"mask.pt")) # mask2 shape (840,840)



        for i, t in enumerate(iterator):
            current_features = {}
            for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                if i <= int(output_block_self_attn_map_injection_threshold):
                    output_q = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt")) # attn: (16, patch**2, dim)
                    output_k = torch.load(os.path.join(source_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                    
                    target_output_q = torch.load(os.path.join(target_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                    target_output_k = torch.load(os.path.join(target_experiment_qkv_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                    
                    # TODO: swap the feature maps
                    # resize the mask, feature to the size of the output_q
                    # for each patch in the output_q, if it is in the mask, find the corresponding patch in the target_output_q and swap them
                    # the corresponding patch is the one with the highest cosine similarity according to the src_feature and tgt_feature
                    swapped_output_q = output_q.clone()
                    swapped_output_k = output_k.clone()

                    patch_size = int(torch.sqrt(torch.tensor(output_q.shape[1])).item())

                    src_feature_reshaped = src_feature.squeeze().permute(1,0).reshape(-1,60,60)
                    src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(patch_size, patch_size), mode='bilinear').squeeze()  # Shape: (768*2, patch, patch)
                    
                    tgt_feature_reshaped = tgt_feature.squeeze().permute(1,0).reshape(-1,60,60)
                    tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped.squeeze().permute(1,0).reshape(-1,60,60), size=(patch_size, patch_size), mode='bilinear').squeeze()  # Shape: (768*2, patch, patch)
                    # Calculate the cosine similarity between src_feature and tgt_feature
                    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
                    similarity_matrix = cosine_similarity(src_feature_upsampled.reshape(src_feature_upsampled.shape[0],-1).permute(1,0), tgt_feature_upsampled.reshape(tgt_feature_upsampled.shape[0],-1).permute(1,0))  # Shape: (patch_size * patch_size, patch_size * patch_size)
                    # Resize the mask and feature to the size of output_q
                    resized_src_mask = F.interpolate(src_mask.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='nearest').squeeze()  # Shape: (patch_size, patch_size)
                    resized_tgt_mask = F.interpolate(tgt_mask.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='nearest').squeeze()  # Shape: (patch_size, patch_size)

                    for patch_idx in range(output_q.shape[1]):
                        # If the patch is in the src_mask, find the corresponding patch in the target_output_q and swap them
                        if resized_src_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
                            # Find the corresponding patch with the highest cosine similarity
                            tgt_patch_idx = similarity_matrix[patch_idx].argmax()

                            # Swap the patches in output_q
                            swapped_output_q[:, patch_idx] = target_output_q[:, tgt_patch_idx]

                            # If swapped_output_k is not None, swap the patches in output_k as well
                            if swapped_output_k is not None:
                                swapped_output_k[:, patch_idx] = target_output_k[:, tgt_patch_idx]

                    current_features[f'output_block_{output_block_idx}_self_attn_q'] = swapped_output_q
                    current_features[f'output_block_{output_block_idx}_self_attn_k'] = swapped_output_k



            for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
                if i <= int(feature_injection_threshold):
                    output = torch.load(os.path.join(source_experiment_out_layers_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))  # Shape: (2, 1280, 16, 16)
                    target_output = torch.load(os.path.join(target_experiment_out_layers_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))  # Shape: (2, 1280, 16, 16)

                    # Resize the mask to the size of output
                    resized_src_mask_out_layers = F.interpolate(src_mask.unsqueeze(0).unsqueeze(0), size=(output.shape[2], output.shape[3]), mode='nearest').squeeze()  # Shape: (16, 16)
                    resized_tgt_mask_out_layers = F.interpolate(tgt_mask.unsqueeze(0).unsqueeze(0), size=(output.shape[2], output.shape[3]), mode='nearest').squeeze()  # Shape: (16, 16)

                    # Calculate the cosine similarity between src_feature and tgt_feature for out_layers
                    src_feature_upsampled_out_layers = F.interpolate(src_feature_reshaped, size=(output.shape[2], output.shape[3]), mode='bilinear').squeeze()  # Shape: (768*2, 16, 16)
                    tgt_feature_upsampled_out_layers = F.interpolate(tgt_feature_reshaped, size=(output.shape[2], output.shape[3]), mode='bilinear').squeeze()  # Shape: (768*2, 16, 16)
                    similarity_matrix_out_layers = cosine_similarity(src_feature_upsampled_out_layers.reshape(src_feature_upsampled_out_layers.shape[0],-1).permute(1,0), tgt_feature_upsampled_out_layers.reshape(tgt_feature_upsampled_out_layers.shape[0],-1).permute(1,0))  # Shape: (16 * 16, 16 * 16)
                    
                    # TODO: swap the feature maps
                    swapped_output = output.clone()

                    for patch_idx in range(output.shape[2] * output.shape[3]):
                        # If the patch is in the resized_src_mask_out_layers, find the corresponding patch in the target_output and swap them
                        if resized_src_mask_out_layers[patch_idx // output.shape[3], patch_idx % output.shape[3]] == 1:
                            # Find the corresponding patch with the highest cosine similarity
                            tgt_patch_idx = similarity_matrix_out_layers[patch_idx].argmax()

                            tgt_patch_row = tgt_patch_idx // output.shape[3]
                            tgt_patch_col = tgt_patch_idx % output.shape[3]

                            # Swap the patches in output
                            swapped_output[:, :, patch_idx // output.shape[3], patch_idx % output.shape[3]] = target_output[:, :, tgt_patch_row, tgt_patch_col]

                    current_features[f'output_block_{output_block_idx}_out_layers'] = swapped_output

            target_features.append(current_features)

        return target_features # for one timestep, attn: (16, patch**2, dim); out_layers: (2, 1280, 16, 16)

    batch_size = len(exp_config.prompts)
    prompts = exp_config.prompts
    assert prompts is not None

    start_code_path = f"{exp_path_root}/{exp_config.source_experiment_name}/z_enc.pt"
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
