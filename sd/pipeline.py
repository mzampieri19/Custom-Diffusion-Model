import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# Image dimensions and latent space dimensions
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    """
    Generates an image from a text prompt (and optionally an input image) using a diffusion model pipeline.
    """
    with torch.no_grad():
        # Validate strength parameter
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # Helper to move models to idle device if specified
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Set up random generator for reproducibility
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # Load and move CLIP model to device
        clip = models["clip"]
        clip.to(device)
        
        # Prepare context embeddings for classifier-free guidance (CFG)
        if do_cfg:
            # Encode conditional prompt
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            # Encode unconditional prompt
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            # Concatenate contexts for CFG
            context = torch.cat([cond_context, uncond_context])
        else:
            # Encode only the conditional prompt
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)

        # Select and configure sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        # Define shape of latent tensor
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            # If an input image is provided, encode it to latent space
            encoder = models["encoder"]
            encoder.to(device)

            # Preprocess input image
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Add noise to the encoded image
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # Otherwise, start from random noise
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # Load and move diffusion model to device
        diffusion = models["diffusion"]
        diffusion.to(device)

        # Diffusion process: denoise latents step by step
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                # Duplicate latents for CFG (conditional and unconditional)
                model_input = model_input.repeat(2, 1, 1, 1)

            # Predict noise with diffusion model
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Apply classifier-free guidance
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Update latents using the sampler
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # Decode latents to image space
        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        # Post-process and return image
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    """
    Rescales tensor x from old_range to new_range.
    Optionally clamps the result to new_range.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    """
    Returns sinusoidal time embedding for a given timestep.
    """
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
