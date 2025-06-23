from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    """
    Loads model weights from a checkpoint and initializes model components.

    Args:
        ckpt_path (str): Path to the checkpoint file containing model weights.
        device (str or torch.device): Device to load the models onto.

    Returns:
        dict: Dictionary containing initialized model components.
    """
    # Load the state dictionary from the checkpoint using the model_converter utility
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # Initialize the VAE encoder and load its weights
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    # Initialize the VAE decoder and load its weights
    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # Initialize the diffusion model and load its weights
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    # Initialize the CLIP model and load its weights
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    # Return all loaded models in a dictionary
    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }