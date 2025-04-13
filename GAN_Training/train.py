import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os

from pix2pix_model import UNetGenerator, PatchDiscriminator

# Add damaged and ground truth images to folders data/damaged and data/clean

class RestorationDataset(Dataset):
    def __init__(self, damaged_dir, clean_dir, transform=None):
        def list_images(folder):
            return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png'))])

        self.damaged_images = list_images(damaged_dir)
        self.clean_images = list_images(clean_dir)
        self.damaged_dir = damaged_dir
        self.clean_dir = clean_dir
        self.transform = transform

    def __len__(self):
        return len(self.damaged_images)

    def __getitem__(self, idx):
        damaged_path = os.path.join(self.damaged_dir, self.damaged_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        damaged = Image.open(damaged_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        original_size = clean.size  # (width, height)
        if self.transform:
            damaged = self.transform(damaged)
            clean = self.transform(clean)
        return damaged, clean, original_size


def train(dataloader, generator, discriminator, optimizerG, optimizerD, bce_loss, l1_loss, device, val_loader, num_epochs=5):

    os.makedirs("outputs/fake", exist_ok=True)
    os.makedirs("outputs/real", exist_ok=True)
    os.makedirs("outputs/eval_fake", exist_ok=True)
    os.makedirs("outputs/eval_real", exist_ok=True)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        lossD = 0.0
        lossG = 0.0
        for i, (damaged, clean, original_size) in enumerate(dataloader):
            damaged = damaged.to(device)
            clean = clean.to(device)

            # Train Discriminator
            fake = generator(damaged)
            disc_real = discriminator(damaged, clean)
            disc_fake = discriminator(damaged, fake.detach())
            lossD_real = bce_loss(disc_real, torch.ones_like(disc_real))
            lossD_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()

            # Train Generator
            disc_fake = discriminator(damaged, fake)
            lossG_adv = bce_loss(disc_fake, torch.ones_like(disc_fake))
            lossG_l1 = l1_loss(fake, clean)
            lossG = lossG_adv + 100 * lossG_l1

            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()

            # Save multiple samples every 20 batches
            if i % 20 == 0:
                for j in range(min(3, fake.size(0))):
                    unsuq_fake = fake[j].cpu().unsqueeze(0)
                    resized_fake = nn.functional.interpolate(unsuq_fake, size=(original_size[1][j], original_size[0][j]), mode='bilinear')
                    save_image(resized_fake, f"outputs/fake/epoch{epoch}_batch{i}_sample{j}.png")
                    unsuq_clean = clean[j].cpu().unsqueeze(0)
                    resized_clean = nn.functional.interpolate(unsuq_clean, size=(original_size[1][j], original_size[0][j]), mode='bilinear')
                    save_image(resized_clean, f"outputs/real/epoch{epoch}_batch{i}_sample{j}.png")

        print(f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

        # Evaluation
        
        generator.eval()

        total_l1_loss = 0.0
        total_batches = 0
        l1_loss_fn = nn.L1Loss()

        with torch.no_grad():
            for i, (damaged, clean, original_size) in enumerate(dataloader):
                damaged = damaged.to(device)
                clean = clean.to(device)
                fake = generator(damaged)

                # Accumulate L1 loss
                batch_l1 = l1_loss_fn(fake, clean).item()
                total_l1_loss += batch_l1
                total_batches += 1

                for j in range(min(3, fake.size(0))):
                    unsuq_fake = fake[j].cpu().unsqueeze(0)
                    resized_fake = nn.functional.interpolate(unsuq_fake, size=(original_size[1][j], original_size[0][j]), mode='bilinear')
                    save_image(resized_fake, f"outputs/eval_fake/epoch{epoch}_sample{i}_{j}.png")
                    unsuq_clean = clean[j].cpu().unsqueeze(0)
                    resized_clean = nn.functional.interpolate(unsuq_clean, size=(original_size[1][j], original_size[0][j]), mode='bilinear')
                    save_image(resized_clean,f"outputs/eval_real/epoch{epoch}_sample{i}_{j}.png")

        avg_l1_loss = total_l1_loss / total_batches
        print(f"Validation L1 Loss: {avg_l1_loss:.4f}")


def evaluate(dataloader, generator, device):
    generator.eval()
    os.makedirs("outputs/eval_fake", exist_ok=True)
    os.makedirs("outputs/eval_real", exist_ok=True)

    total_l1_loss = 0.0
    total_batches = 0
    l1_loss_fn = nn.L1Loss()

    with torch.no_grad():
        for i, (damaged, clean, original_size) in enumerate(val_loader):
            damaged = damaged.to(device)
            clean = clean.to(device)
            fake = generator(damaged)

            # Accumulate L1 loss
            batch_l1 = l1_loss_fn(fake, clean).item()
            total_l1_loss += batch_l1
            total_batches += 1

            for j in range(min(3, fake.size(0))):
                unsuq_fake = fake[j].cpu().unsqueeze(0)
                resized_fake = nn.functional.interpolate(unsuq_fake, size=(original_size[1][j], original_size[0][j]), mode='bilinear')
                save_image(resized_fake, f"outputs/eval_fake/sample{i}_{j}.png")
                unsuq_clean = clean[j].cpu().unsqueeze(0)
                resized_clean = nn.functional.interpolate(unsuq_clean, size=(original_size[1][j], original_size[0][j]), mode='bilinear')
                save_image(resized_clean,f"outputs/eval_real/sample{i}_{j}.png")

            if i >= 2:
                break  # Save only a few batches for evaluation preview

    avg_l1_loss = total_l1_loss / total_batches
    print(f"Validation L1 Loss: {avg_l1_loss:.4f}")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = RestorationDataset("data/damaged", "data/clean", transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    train(train_loader, generator, discriminator, optimizerG, optimizerD, bce_loss, l1_loss, device, val_loader)
