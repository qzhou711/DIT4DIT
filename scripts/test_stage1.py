"""Simple test script for Stage 1: predict future images from a checkpoint.

Usage:
    python scripts/test_stage1.py \
    --suite libero_object \
    --checkpoint ./checkpoints/libero_object/stage1/step_1000 \
    --cosmos_model_id ./checkpoints/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
    --device cuda --ode_steps 10
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig, get_suite_data_config, LIBERO_SUITES
from mimic_video.data.dataset import MimicVideoDataset
from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.flow_matching import FlowMatchingScheduler


def main():
    parser = argparse.ArgumentParser(description="Test Stage 1: future image prediction")
    parser.add_argument("--suite", type=str, required=True, choices=list(LIBERO_SUITES.keys()))
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 1 LoRA checkpoint")
    parser.add_argument("--cosmos_model_id", type=str, required=True, help="Path to Cosmos model")
    parser.add_argument("--output_path", type=str, default="stage1_test.gif", help="Output GIF path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ode_steps", type=int, default=20, help="Euler ODE steps for denosing")
    args = parser.parse_args()

    device = args.device
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    
    print(f"Using device: {device}")

    # 1. Config
    data_config = get_suite_data_config(args.suite)
    model_config = ModelConfig()
    model_config.cosmos_model_id = args.cosmos_model_id

    # 2. Dataset
    print(f"Loading dataset for {args.suite}...")
    # Use only 1 episode for testing
    test_dataset = MimicVideoDataset(
        repo_id=data_config.repo_id,
        camera_names=data_config.camera_names,
        state_keys=data_config.state_keys,
        action_keys=data_config.action_keys,
        num_pixel_frames=data_config.num_pixel_frames,
        action_chunk_size=data_config.action_chunk_size,
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        target_height=data_config.camera_height,
        target_width=data_config.camera_width,
        episode_indices=[0],  # just the first episode
        precomputed_dir=data_config.precomputed_dir,
        action_norm_type=data_config.action_norm_type,
        fps=data_config.fps,
    )
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    
    # 3. Model
    print("Initializing Cosmos video backbone...")
    backbone = CosmosVideoBackbone(
        model_id=model_config.cosmos_model_id,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_target_modules=model_config.lora_target_modules,
        hidden_state_layer=model_config.hidden_state_layer,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device=device,
    )
    
    print(f"Loading LoRA from {args.checkpoint}...")
    backbone.load_lora(args.checkpoint, is_trainable=False)
    backbone.transformer.to(device)
    backbone.transformer.eval()
    backbone.offload_vae_and_text_encoder("cpu")
    
    # 4. Inference setup
    fm = FlowMatchingScheduler()
    num_cond_latent_frames = data_config.num_cond_latent_frames
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Prepare inputs
    video = batch["video"][:1]  # [1, T, C, H, W]
    video = video.permute(0, 2, 1, 3, 4).to(device)  # [1, C, T, H, W]

    with torch.no_grad():
        # Encode full video to latents
        z_0 = backbone.encode_video(video)  # [1, C_lat, T_lat, H_lat, W_lat]
        
        z_cond = z_0[:, :, :num_cond_latent_frames]
        z_pred_gt = z_0[:, :, num_cond_latent_frames:]
        
        # Get T5 embedding
        if "t5_embedding" in batch:
            t5_emb = batch["t5_embedding"][:1].to(device, dtype=compute_dtype)
            if t5_emb.ndim == 4 and t5_emb.shape[1] == 1:
                t5_emb = t5_emb.squeeze(1)
        else:
            # Try to load single-task t5 embedding if missing in batch
            single_t5_path = os.path.join(data_config.precomputed_dir, "t5_embedding.pt")
            if os.path.exists(single_t5_path):
                t5_emb = torch.load(single_t5_path, map_location=device, weights_only=True).to(dtype=compute_dtype)
                t5_emb = t5_emb.expand(1, -1, -1)
            else:
                raise ValueError("T5 embedding not found. Run precompute_embeddings.py first.")

        # Inference loop (Euler ODE)
        print(f"Running inference with {args.ode_steps} ODE steps...")
        z_noise = torch.randn_like(z_pred_gt)

        def model_fn(z_t, tau):
            tau_tensor = torch.tensor([tau], device=z_t.device, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=compute_dtype):
                _, full_out = backbone.forward_transformer(
                    z_noisy=z_t,
                    z_cond=z_cond,
                    tau_v=tau_tensor,
                    encoder_hidden_states=t5_emb,
                )
            T_cond = num_cond_latent_frames
            x0_pred = full_out[:, :, T_cond:]
            return (z_t - x0_pred) / max(tau, 1e-6)

        z_pred_denoised = fm.ode_solve_euler(
            model_fn, z_noise, num_steps=args.ode_steps, tau_start=1.0, tau_end=0.0
        )

        # 5. Decode and save
        print("Decoding predicted latents...")
        gt_full = backbone.decode_video(z_0)  # [1, C, T, H, W]
        pred_latents = torch.cat([z_cond, z_pred_denoised], dim=2)
        pred_full = backbone.decode_video(pred_latents)  # [1, C, T, H, W]

    # Convert to uint8 numpy: [T, H, W, C]
    def to_video_np(x):
        x = (x.squeeze(0).permute(1, 2, 3, 0).clamp(-1, 1) * 0.5 + 0.5) * 255
        return x.cpu().to(torch.uint8).numpy()

    gt_np = to_video_np(gt_full)
    pred_np = to_video_np(pred_full)

    # Side-by-side
    side_by_side = np.concatenate([gt_np, pred_np], axis=2)  # [T, H, 2*W, C]

    print(f"Saving GIF to {args.output_path}...")
    pil_frames = [Image.fromarray(f) for f in side_by_side]
    pil_frames[0].save(
        args.output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # 10 FPS
        loop=0
    )
    print("Done!")

if __name__ == "__main__":
    main()
