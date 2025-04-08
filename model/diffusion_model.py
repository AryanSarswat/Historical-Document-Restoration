import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm

def get_unet_model(image_size=64):
    """
    Initializes and returns a UNet2DModel from the diffusers library.
    """
    # Configuration for the U-Net. Adjust these based on experimentation.
    # More blocks/channels generally means higher capacity but more memory/compute.
    model = UNet2DModel(
        sample_size=image_size,           # Target image resolution
        in_channels=3,                    # Input channels (RGB)
        out_channels=3,                   # Output channels (RGB) - predicting noise of same shape
        layers_per_block=2,               # Number of ResNet blocks per level
        block_out_channels=(128, 128, 256, 256, 512, 512), # Down/Up block channels
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D", # Add attention in deeper layers
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",   # Add attention in deeper layers
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

class DiffusionRestorationModel(nn.Module):
    """
    Encapsulates the U-Net model, noise scheduler, and diffusion sampling logic
    for image restoration/generation tasks.
    """
    def __init__(self, image_size=64, in_channels=3, out_channels=3, device='cpu'):
        """
        Initializes the Diffusion Restoration Model.

        Args:
            image_size (int): The size of the input/output images.
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output image channels (usually same as input).
            unet_config (dict, optional): Arguments to pass to UNet2DModel constructor.
                                           Overrides default get_unet_model call if provided.
            noise_scheduler_config (dict, optional): Arguments for DDPMScheduler.
                                                     Defaults: num_train_timesteps=1000, beta_schedule='linear'.
            device (torch.device or str): Device to run the model on.
        """
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        # Initialize U-Net
        self.unet = get_unet_model(image_size=image_size).to(device)

        # Initialize Noise Scheduler
        noise_scheduler_config = {'num_train_timesteps': 1000, 'beta_schedule': 'linear'}
        self.noise_scheduler = DDPMScheduler(**noise_scheduler_config)

    def forward(self, noisy_images, timesteps):
        """
        Performs the forward pass through the U-Net to predict noise.

        Args:
            noisy_images (Tensor): Batch of noisy images (B, C, H, W).
            timesteps (Tensor): Batch of timesteps (B,).

        Returns:
            Tensor: Predicted noise (same shape as noisy_images).
        """
        return self.unet(noisy_images, timesteps).sample

    def add_noise(self, original_images, noise, timesteps):
        """Adds noise to images using the scheduler."""
        return self.noise_scheduler.add_noise(original_images, noise, timesteps)

    def predict_start_from_noise(self, noisy_images, t, noise_pred):
        """
        Predicts the original image (x0) from a noisy image and predicted noise.
        Uses the scheduler's formula.

        Args:
            noisy_images (Tensor): The noisy image at timestep t (xt).
            t (Tensor or int): The timestep(s).
            noise_pred (Tensor): The predicted noise (epsilon_theta).

        Returns:
            Tensor: The predicted original image (pred_x0).
        """
        # Ensure t is on the correct device if it's a tensor
        if isinstance(t, torch.Tensor):
            t = t.to(self.device)

        # Use scheduler's internal alpha values
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        # Handle single timestep or batch of timesteps
        if isinstance(t, int):
            sqrt_alpha_bar_t = alphas_cumprod[t].sqrt()
            sqrt_one_minus_alpha_bar_t = (1.0 - alphas_cumprod[t]).sqrt()
        else: # Assuming t is a tensor of timesteps for the batch
             sqrt_alpha_bar_t = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
             sqrt_one_minus_alpha_bar_t = (1.0 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)

        pred_x0 = (noisy_images - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        return pred_x0

    @torch.no_grad() # Sampling should not compute gradients
    def sample(self, start_image, num_inference_steps=None):
        """
        Performs the reverse diffusion process (sampling) starting from an image.

        Args:
            start_image (Tensor): The image to start the reverse process from
                                  (e.g., pure noise or a damaged image). Shape (B, C, H, W).
            num_inference_steps (int, optional): Number of diffusion steps for sampling.
                                                 Defaults to scheduler's train steps.
        Returns:
            Tensor: The generated/restored image.
        """
        self.unet.eval() # Set U-Net to evaluation mode

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps

        # Set inference timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        image = start_image.to(self.device)

        for t in tqdm(timesteps, desc="Diffusion Sampling"):
            # Predict noise residual
            t = t.to(self.device)
            
            noise_pred = self.forward(image, t)

            # Compute previous noisy sample x_t -> x_{t-1}
            image = self.noise_scheduler.step(noise_pred, t, image).prev_sample

        self.unet.train()
        return image.clamp(-1, 1) 

    def save_checkpoint(self, path, epoch, optimizer):
        """
        Saves the U-Net model state and optimizer state.

        Args:
            path (str): Path to save the checkpoint file.
            epoch (int): Current epoch number.
            optimizer (torch.optim.Optimizer): Model optimizer.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Diffusion Model Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None):
        """
        Loads the U-Net model state and optionally optimizer state.

        Args:
            path (str): Path to the checkpoint file.
            optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.

        Returns:
            int: The epoch number to resume training from (epoch saved + 1).
                 Returns 0 if checkpoint not found or epoch key missing.
            dict: The loaded checkpoint dictionary (or None if file not found).
        """
        if not os.path.isfile(path):
            print(f"Checkpoint file not found: {path}")
            return 0, None

        print(f"Loading Diffusion Model checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.unet.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state.")
            except ValueError as e:
                 print(f"Warning: Could not load optimizer state, possibly due to parameter mismatch: {e}")
                 print("Optimizer state will be reset.")


        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, checkpoint

if __name__ == '__main__':
    # Example Usage & Testing
    print("Testing Diffusion Restoration Model Class...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 64 # Use smaller size for faster testing
    model = DiffusionRestorationModel(image_size=img_size, in_channels=3, out_channels=3, device=device)

    print("\nTesting Forward Pass...")
    batch_size = 2
    dummy_noisy_image = torch.randn(batch_size, 3, img_size, img_size, device=device)
    dummy_timestep = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
    noise_pred = model.forward(dummy_noisy_image, dummy_timestep)
    print("Input noisy shape:", dummy_noisy_image.shape)
    print("Input timestep shape:", dummy_timestep.shape)
    print("Output noise_pred shape:", noise_pred.shape)
    assert noise_pred.shape == dummy_noisy_image.shape
    print("Forward pass test successful.")

    print("\nTesting Sampling...")
    start_noise = torch.randn(1, 3, img_size, img_size, device=device) # Start from noise
    generated_image = model.sample(start_noise, num_inference_steps=10) # Fewer steps for quick test
    print("Start noise shape:", start_noise.shape)
    print("Generated image shape:", generated_image.shape)
    assert generated_image.shape == start_noise.shape
    print("Sampling test successful.")


    # Test save/load
    opt = torch.optim.Adam(model.unet.parameters())
    test_path = "test_diffusion_checkpoint.pth"
    print("\nTesting Save Checkpoint...")
    model.save_checkpoint(test_path, epoch=3, optimizer=opt)

    print("\nTesting Load Checkpoint...")
    new_model = DiffusionRestorationModel(image_size=img_size, in_channels=3, out_channels=3, device=device)
    new_opt = torch.optim.Adam(new_model.unet.parameters())
    start_epoch, loaded_data = new_model.load_checkpoint(test_path, new_opt)

    assert start_epoch == 4
    assert torch.equal(list(model.unet.parameters())[0].data, list(new_model.unet.parameters())[0].data)
    print("Save/Load test successful.")

    # Clean up test file
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"Removed test file: {test_path}")
