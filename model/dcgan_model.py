import os
import torch
import torch.nn as nn

def weights_init(m):
    """
    Initializes weights for Conv and BatchNorm layers according to DCGAN paper.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    DCGAN Generator Network.
    Takes a latent vector z and outputs an image.
    """
    def __init__(self, dim=100, features=64, num_channels=3, image_size=64):
        """
        Args:
            dim (int): Size of the latent z vector.
            features (int): Size of feature maps in generator.
            num_channels (int): Number of channels in the output image.
            image_size (int): Target output image size (e.g., 64).
                               Architecture assumes powers of 2.
        """
        super(Generator, self).__init__()
        self.dim = dim
        self.features = features
        self.num_channels = num_channels
        self.image_size = image_size

        num_layers = int(torch.log2(torch.tensor(float(image_size)))) - 3
        if image_size != 2**(num_layers + 3):
             print(f"Warning: image_size {image_size} is not a power of 2. Architecture might not scale correctly.")

        layers = []
        layers.append(nn.ConvTranspose2d(dim, features * (2**num_layers), 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(features * (2**num_layers)))
        layers.append(nn.ReLU(True))

        # Intermediate layers: Upsample to image_size / 2
        for i in range(num_layers):
            in_feat = features * (2**(num_layers - i))
            out_feat = features * (2**(num_layers - i - 1))
            layers.append(nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(True))

        # Output layer: features -> num_channels x image_size x image_size
        layers.append(nn.ConvTranspose2d(features, num_channels, 4, 2, 1, bias=False))
        layers.append(nn.Tanh()) # Output range [-1, 1]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass through the generator.
        Args:
            input (Tensor): Latent vector z, shape (batch_size, dim, 1, 1).
        Returns:
            Tensor: Generated image, shape (batch_size, num_channels, image_size, image_size).
        """
        return self.layers(input)

class Discriminator(nn.Module):
    """
    DCGAN Discriminator Network.
    Takes an image and outputs a probability (real/fake).
    """
    def __init__(self, num_channels=3, features=64, image_size=64):
        """
        Args:
            num_channels (int): Number of channels in the input image.
            features (int): Size of feature maps in discriminator.
            image_size (int): Input image size (e.g., 64).
                               Architecture assumes powers of 2.
        """
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.features = features
        self.image_size = image_size

        num_layers = int(torch.log2(torch.tensor(float(image_size)))) - 3
        if image_size != 2**(num_layers + 3):
             print(f"Warning: image_size {image_size} is not a power of 2. Architecture might not scale correctly.")


        layers = []
        # Input layer: num_channels x image_size x image_size -> features x image_size/2 x image_size/2
        layers.append(nn.Conv2d(num_channels, features, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers: Downsample to 4x4 feature map
        for i in range(num_layers):
            in_feat = features * (2**i)
            out_feat = features * (2**(i + 1))
            layers.append(nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output layer: (features * 2**num_layers) x 4 x 4 -> 1 x 1 x 1
        layers.append(nn.Conv2d(features * (2**num_layers), 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass through the discriminator.
        Args:
            input (Tensor): Input image, shape (batch_size, num_channels, image_size, image_size).
        Returns:
            Tensor: Probability score (scalar per image), shape (batch_size,).
        """
        return self.main(input).view(-1)

class DCGAN:
    """
    Encapsulates the DCGAN model (Generator and Discriminator)
    and provides methods for saving and loading checkpoints.
    """
    def __init__(self, dim=100, features_d=64, features_g=64, num_channels=3, image_size=64, device='cpu', ngpu=1):
        """
        Initializes the DCGAN model.

        Args:
            dim (int): Size of latent vector z.
            features (int): Feature maps in Generator.
            features (int): Feature maps in Discriminator.
            num_channels (int): Image channels.
            image_size (int): Image size.
            device (torch.device or str): Device to run the models on.
            ngpu (int): Number of GPUs to use (for DataParallel).
        """
        self.dim = dim
        self.features_discriminator = features_d
        self.features_generator = features_g
        self.num_channels = num_channels
        self.image_size = image_size
        self.device = device
        self.ngpu = ngpu

        # Initialize Generator and Discriminator
        self.netG = Generator(dim=dim, features=self.features_discriminator, num_channels=num_channels, image_size=image_size).to(device)
        self.netD = Discriminator(num_channels=num_channels, features=self.features_generator, image_size=image_size).to(device)

        # Apply weight initialization
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # Handle multi-gpu if desired
        if device.type == 'cuda' and ngpu > 1:
            print(f"Using DataParallel for DCGAN on {ngpu} GPUs.")
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

    def get_models(self):
        """Returns the generator and discriminator models."""
        return self.netG, self.netD

    def save_checkpoint(self, path, epoch, optimizerG, optimizerD, fixed_noise=None, lossesG=None, lossesD=None, args=None):
        """
        Saves the model and optimizer states to a checkpoint file.

        Args:
            path (str): Path to save the checkpoint file.
            epoch (int): Current epoch number.
            optimizerG (torch.optim.Optimizer): Generator optimizer.
            optimizerD (torch.optim.Optimizer): Discriminator optimizer.
            fixed_noise (Tensor, optional): Fixed noise vector for visualization.
            lossesG (list, optional): History of Generator losses.
            lossesD (list, optional): History of Discriminator losses.
            args (argparse.Namespace, optional): Training arguments.
        """
        # Get state dicts, handling DataParallel if necessary
        netG_state_dict = self.netG.module.state_dict() if isinstance(self.netG, nn.DataParallel) else self.netG.state_dict()
        netD_state_dict = self.netD.module.state_dict() if isinstance(self.netD, nn.DataParallel) else self.netD.state_dict()

        checkpoint = {
            'epoch': epoch,
            'modelG_state_dict': netG_state_dict,
            'modelD_state_dict': netD_state_dict,
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'fixed_noise': fixed_noise,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'args': args
        }
        torch.save(checkpoint, path)
        print(f"DCGAN Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizerG=None, optimizerD=None):
        """
        Loads model and optimizer states from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
            optimizerG (torch.optim.Optimizer, optional): Generator optimizer to load state into.
            optimizerD (torch.optim.Optimizer, optional): Discriminator optimizer to load state into.

        Returns:
            int: The epoch number to resume training from (epoch saved + 1).
                 Returns 0 if checkpoint not found or epoch key missing.
            dict: The loaded checkpoint dictionary (or None if file not found).
        """
        if not os.path.isfile(path):
            print(f"Checkpoint file not found: {path}")
            return 0, None

        print(f"Loading DCGAN checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state dicts, handling DataParallel if necessary
        netG_model = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        netD_model = self.netD.module if isinstance(self.netD, nn.DataParallel) else self.netD

        netG_model.load_state_dict(checkpoint['modelG_state_dict'])
        netD_model.load_state_dict(checkpoint['modelD_state_dict'])

        # Load optimizer states if provided
        if optimizerG and 'optimizerG_state_dict' in checkpoint:
            optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
            print("Loaded Generator optimizer state.")
        if optimizerD and 'optimizerD_state_dict' in checkpoint:
            optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
            print("Loaded Discriminator optimizer state.")

        start_epoch = checkpoint.get('epoch', -1) + 1 # Start from next epoch
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, checkpoint


if __name__ == '__main__':
    # Example Usage & Testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCGAN(device=device, image_size=32) # Test with smaller size

    netG, netD = model.get_models()

    # Test save/load (requires dummy optimizers)
    optG = torch.optim.Adam(netG.parameters())
    optD = torch.optim.Adam(netD.parameters())
    test_path = "test_dcgan_checkpoint.pth"

    model.save_checkpoint(test_path, epoch=5, optimizerG=optG, optimizerD=optD)

    # Create a new model instance to load into
    new_model = DCGAN(device=device, image_size=32)
    new_optG = torch.optim.Adam(new_model.netG.parameters())
    new_optD = torch.optim.Adam(new_model.netD.parameters())
    start_epoch, loaded_data = new_model.load_checkpoint(test_path, new_optG, new_optD)

    assert start_epoch == 6
    # Basic check: compare a parameter from loaded vs original
    assert torch.equal(list(model.netG.parameters())[0], list(new_model.netG.parameters())[0])

    # Clean up test file
    if os.path.exists(test_path):
        os.remove(test_path)
