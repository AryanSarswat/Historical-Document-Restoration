# baseline_with_ocr_eval.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import argparse
from tqdm import tqdm
import wandb
import random
from jiwer import cer, wer

# Import models and utilities from other files
from model.pix2pix_model import UNetGenerator, PatchDiscriminator
from trocr_utils import load_trocr_model # Import TrOCR loading utility
# Use the dataset that provides text labels for evaluation
from dataset.restoration_dataset import RestorationDatasetWithText

# --- Configuration ---
def parse_args():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Baseline Pix2Pix GAN Training with OCR Evaluation")
    # GAN params
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training/evaluation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizers")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizers")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight for L1 loss in Generator")
    parser.add_argument("--img_size", type=int, default=512, help="Size to resize images to (must be square)")
    # Data params
    parser.add_argument("--damaged_dir", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/damaged", help="Directory for damaged images")
    parser.add_argument("--clean_dir", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/data/line_images_normalized", help="Directory for clean (ground truth) images")
    parser.add_argument("--transcription_file", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/ground_truth/transcription.txt", help="Directory for ground transcriptions")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline_ocr_eval_imgsize_512", help="Directory to save outputs")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 4, help="Number of dataloader workers")
    # Logging/Saving params
    parser.add_argument("--log_interval", type=int, default=50, help="Log training stats every N batches")
    parser.add_argument("--save_img_interval", type=int, default=5, help="Save sample images every N epochs")
    parser.add_argument("--wandb_project", type=str, default="HUM_Project", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)") # Optional
    # TrOCR params (for evaluation only)
    parser.add_argument("--trocr_model_name", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/finetuned_trocr", help="Name of the TrOCR model for evaluation")
    parser.add_argument("--max_text_length", type=int, default=128, help="Max sequence length for TrOCR tokenizer")

    return parser.parse_args()

# --- Utility Functions (Copied/Adapted from baseline.py) ---
def set_seed(seed: int):
    """ Sets random seed for reproducibility. """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU

def denormalize(tensor: torch.Tensor):
    """ Denormalizes a tensor from [-1, 1] to [0, 1]. """
    return tensor * 0.5 + 0.5

def save_sample_images(epoch: int, loader: DataLoader, generator: nn.Module, device: torch.device, output_dir: str, wandb_log: bool = True, num_samples: int = 3):
    """ Saves and logs sample input/output/target images (without OCR text). """
    generator.eval()
    # Get a batch - use next(iter()) carefully, ensure loader is not exhausted if called multiple times
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Warning: Validation loader exhausted, cannot save sample images.")
        generator.train()
        return

    if batch is None or batch[0] is None:
        print("Warning: Skipping sample image saving due to corrupted batch.")
        generator.train()
        return

    # Unpack, ignoring text labels here
    damaged, clean, _, damaged_fnames, clean_fnames, (original_width, original_height) = batch
    damaged, clean = damaged.to(device), clean.to(device)

    with torch.no_grad():
        fake = generator(damaged)

    # Take only num_samples
    random_idx = torch.randperm(damaged.size(0))[:num_samples] 
    damaged = damaged[random_idx]
    clean = clean[random_idx]
    fake = fake[random_idx]
    original_width = original_width[random_idx]
    original_height = original_height[random_idx]
    original_size = list(zip(original_height, original_width))

    # Denormalize for saving/viewing
    damaged = denormalize(damaged)
    clean = denormalize(clean)
    fake = denormalize(fake)

    # Resize
    def resize_batch(imgs, sizes):
        out = [F.interpolate(img.unsqueeze(0),
                                 size=tuple(sz),
                                 mode="bilinear",
                                 align_corners=False).squeeze(0)
                   for img, sz in zip(imgs, sizes)]
        return out

    damaged = resize_batch(damaged, original_size)
    fake    = resize_batch(fake,    original_size)
    clean   = resize_batch(clean,   original_size)

    # Concatenate images: Damaged | Fake | Clean
    sample_rows = []
    wandb_images = []

    for idx in range(num_samples):
        row_grid = make_grid(                             # C × H × (3W)
            torch.stack([damaged[idx],
                         fake[idx],
                         clean[idx]]),
            nrow=1, padding=3, normalize=True
        )
        sample_rows.append(row_grid)

        if wandb_log:
            wandb_images.append(
                wandb.Image(row_grid, caption=f"Epoch {epoch} - {damaged_fnames [idx]}")
            )

    # Save grid
    for idx, g in enumerate(sample_rows):
        out_f = os.path.join(output_dir, f"epoch_{epoch:03d}_sample_{idx}.png")
        save_image(g, out_f)

    if wandb_log:
        try:
            # Also log the grid
            wandb.log({"Sample Grid": wandb_images})
        except Exception as e:
            print(f"Warning: Failed to log images to wandb: {e}")

    generator.train() # Set back to train mode


# --- Training Loop (Unchanged from original baseline) ---
def train_one_epoch(epoch: int, dataloader: DataLoader, generator: nn.Module, discriminator: nn.Module,
                    optimizerG: optim.Optimizer, optimizerD: optim.Optimizer,
                    bce_loss: nn.Module, l1_loss: nn.Module, lambda_l1: float, device: torch.device,
                    log_interval: int, args: argparse.Namespace):
    """ Trains the GAN for one epoch (no OCR involved). """
    generator.train()
    discriminator.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
    total_loss_d = 0.0
    total_loss_g = 0.0
    total_loss_g_adv = 0.0
    total_loss_g_l1 = 0.0

    for i, batch in enumerate(pbar):
        if batch is None or batch[0] is None:
             print(f"Warning: Skipping corrupted training batch {i}")
             continue

        # Unpack batch data - ignore text labels during training
        damaged, clean, _, _, _, _ = batch
        damaged = damaged.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        # --- Train Discriminator ---
        optimizerD.zero_grad()
        fake_detached = generator(damaged).detach()
        disc_real_output = discriminator(damaged, clean)
        disc_fake_output = discriminator(damaged, fake_detached)
        real_labels = torch.ones_like(disc_real_output, device=device)
        fake_labels = torch.zeros_like(disc_fake_output, device=device)
        lossD_real = bce_loss(disc_real_output, real_labels)
        lossD_fake = bce_loss(disc_fake_output, fake_labels)
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward()
        optimizerD.step()

        # --- Train Generator ---
        optimizerG.zero_grad()
        fake = generator(damaged) # Keep attached for generator update
        disc_fake_output_for_g = discriminator(damaged, fake)
        lossG_adv = bce_loss(disc_fake_output_for_g, real_labels) # Use real labels
        lossG_l1 = l1_loss(fake, clean)
        lossG = lossG_adv + lambda_l1 * lossG_l1
        lossG.backward()
        optimizerG.step()

        # --- Logging ---
        loss_d_item = lossD.item()
        loss_g_item = lossG.item()
        loss_g_adv_item = lossG_adv.item()
        loss_g_l1_item = lossG_l1.item()

        total_loss_d += loss_d_item
        total_loss_g += loss_g_item
        total_loss_g_adv += loss_g_adv_item
        total_loss_g_l1 += loss_g_l1_item

        pbar.set_postfix({
            "Loss_D": f"{loss_d_item:.4f}",
            "Loss_G": f"{loss_g_item:.4f}",
            "G_Adv": f"{loss_g_adv_item:.4f}",
            "G_L1": f"{loss_g_l1_item:.4f}"
        })

        # Log to wandb periodically
        if i % log_interval == 0:
            step = epoch * len(dataloader) + i
            wandb.log({
                "Train/Loss_D": loss_d_item,
                "Train/Loss_G": loss_g_item,
                "Train/Loss_G_Adversarial": loss_g_adv_item,
                "Train/Loss_G_L1": loss_g_l1_item,
                "LearningRate/Generator": optimizerG.param_groups[0]['lr'],
                "LearningRate/Discriminator": optimizerD.param_groups[0]['lr'],
            })

    avg_loss_d = total_loss_d / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_g = total_loss_g / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_g_adv = total_loss_g_adv / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_g_l1 = total_loss_g_l1 / len(dataloader) if len(dataloader) > 0 else 0

    return avg_loss_d, avg_loss_g, avg_loss_g_adv, avg_loss_g_l1

# --- Evaluation (Modified to include OCR metrics) ---
@torch.no_grad() # Disable gradient calculations for evaluation
def evaluate(epoch: int, loader: DataLoader, generator: nn.Module,
             trocr_model: nn.Module, trocr_processor: any, # Add TrOCR components
             l1_loss: nn.Module, device: torch.device):
    """ Evaluates the generator on the validation set, calculating L1 loss, CER, and WER. """
    generator.eval() # Set GAN generator to evaluation mode
    trocr_model.eval() # Ensure TrOCR is in eval mode

    total_l1_loss = 0.0
    all_pred_texts = []
    all_gt_texts = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Evaluating]")

    for batch in pbar:
        if batch is None or batch[0] is None:
             print(f"Warning: Skipping corrupted validation batch")
             continue

        # Unpack batch data (includes text labels)
        damaged, clean, text_labels, _, _, (original_width, original_height) = batch
        damaged = damaged.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        text_labels = text_labels.to(device) # Shape: [batch_size, seq_len]

        # --- L1 Loss Calculation ---
        fake = generator(damaged)
        batch_l1 = l1_loss(fake, clean).item()
        total_l1_loss += batch_l1

        # --- OCR Evaluation ---
        # Denormalize fake images for TrOCR processor
        fake_denorm = denormalize(fake).cpu()

        # -- Resize --
        original_size = list(zip(original_height, original_width))
        def resize_batch(imgs, sizes):
            out = [F.interpolate(img.unsqueeze(0),
                                 size=tuple(sz),
                                 mode="bilinear",
                                 align_corners=False).squeeze(0)
                   for img, sz in zip(imgs, sizes)]
            return out
        fake_denorm = resize_batch(fake_denorm, original_size)

        # Prepare input for TrOCR processor
        pixel_values = trocr_processor(images=list(fake_denorm), return_tensors="pt", do_rescale=False).pixel_values.to(device)

        # Generate text using the TrOCR model
        # Adjust generation parameters if needed (e.g., num_beams)
        generated_ids = trocr_model.generate(pixel_values, max_length=trocr_processor.tokenizer.model_max_length)

        # Decode generated IDs to text
        batch_pred_texts = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_pred_texts.extend(batch_pred_texts)

        # Decode ground truth labels (handle -100 padding)
        batch_gt_texts = []
        for label_tensor in text_labels:
             label_tensor_cpu = label_tensor.cpu()
             label_tensor_cpu[label_tensor_cpu == -100] = trocr_processor.tokenizer.pad_token_id
             decoded = trocr_processor.tokenizer.decode(label_tensor_cpu, skip_special_tokens=True)
             batch_gt_texts.append(decoded)
        all_gt_texts.extend(batch_gt_texts)

        pbar.set_postfix({"Val_L1": f"{batch_l1:.4f}"})

    # --- Calculate Overall Metrics ---
    avg_l1_loss = total_l1_loss / len(loader) if len(loader) > 0 else 0

    # Calculate CER and WER using jiwer
    avg_cer = 0.0
    avg_wer = 0.0
    if all_gt_texts and all_pred_texts: # Ensure lists are not empty
        try:
            # jiwer.compute_measures handles empty strings correctly
            avg_wer = wer(all_gt_texts, all_pred_texts)
            avg_cer = cer(all_gt_texts, all_pred_texts)
            print(f"Epoch {epoch+1} Validation L1 Loss: {avg_l1_loss:.4f}, CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")
        except Exception as e:
            print(f"Error calculating CER/WER with jiwer: {e}")
            print("Ground truth samples:", all_gt_texts[:5])
            print("Prediction samples:", all_pred_texts[:5])
    else:
         print(f"Epoch {epoch+1} Validation L1 Loss: {avg_l1_loss:.4f}. CER/WER calculation skipped (no data).")


    # Log metrics to wandb
    wandb.log({
        "Validation/L1_Loss": avg_l1_loss,
        "Validation/CER": avg_cer,
        "Validation/WER": avg_wer
        }) # Log per epoch

    generator.train() # Set generator back to training mode
    return avg_l1_loss, avg_cer, avg_wer


# --- Main Execution ---
def main():
    args = parse_args()
    set_seed(args.seed)

    # --- Setup Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Setup Wandb ---
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
    print(f"Wandb run URL: {wandb.run.get_url()}")

    # --- Create Directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    # --- Load TrOCR Model (Frozen, for Evaluation Only) ---
    print("Loading TrOCR model for evaluation...")
    trocr_model, trocr_processor = load_trocr_model(args.trocr_model_name, device=device)
    if trocr_model is None or trocr_processor is None:
        print("Failed to load TrOCR model. Evaluation metrics (CER/WER) will not be available.")
        # Allow continuation without OCR eval if loading fails? Or exit? Let's allow continuation.
        # return # Uncomment to exit if TrOCR loading fails
    else:
        # Freeze TrOCR model parameters (important!)
        for param in trocr_model.parameters():
            param.requires_grad = False
        trocr_model.eval() # Set to evaluation mode
        print("TrOCR model loaded and frozen.")

    # --- Data Loading ---
    # Image transforms (same as baseline)
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3), # Normalize to [-1, 1] for GAN
    ])

    try:
        # Use the dataset that includes text processing FOR VALIDATION
        # Training loader can technically use the simpler dataset if memory is tight,
        # but using the same dataset class simplifies the code.
        full_dataset = RestorationDatasetWithText(
            args.damaged_dir,
            args.clean_dir,
            args.transcription_file,
            transform=img_transform,
            text_processor=trocr_processor, # Pass processor for tokenization needed in eval
            max_text_length=args.max_text_length
        )
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return
    except FileNotFoundError as e:
         print(f"Error: Data directory not found - {e}. Please ensure '{args.damaged_dir}' and '{args.clean_dir}' exist.")
         print("Also ensure your text labels are correctly set up for the validation set.")
         return
    except Exception as e:
        print(f"An unexpected error occurred during dataset initialization: {e}")
        return


    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size < 0:
        print(f"Error: Invalid train/validation split sizes ({train_size}/{val_size}) for dataset length {len(full_dataset)}.")
        return
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    print(f"Dataset size: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # drop_last=True for training might prevent issues if the last batch is smaller and causes dimension mismatches
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # No shuffle for validation loader to get consistent samples & eval metrics
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Initialize GAN Models ---
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)

    # --- Initialize Optimizers and Loss Functions ---
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    bce_loss = nn.BCELoss() # Discriminator uses Sigmoid
    l1_loss = nn.L1Loss()

    # Watch GAN models with wandb
    wandb.watch(generator, log="gradients", log_freq=500, idx=0)
    wandb.watch(discriminator, log="gradients", log_freq=500, idx=1)

    # --- Training Loop ---
    print("Starting Training (with OCR evaluation)...")
    best_val_loss = float('inf') # Track best L1 loss

    for epoch in range(args.epochs):
        # --- Training Step ---
        avg_loss_d, avg_loss_g, avg_loss_g_adv, avg_loss_g_l1 = train_one_epoch(
            epoch, train_loader, generator, discriminator, optimizerG, optimizerD,
            bce_loss, l1_loss, args.lambda_l1, device, args.log_interval, args
        )

        # Log average training losses for the epoch
        wandb.log({
            "Epoch": epoch + 1,
            "Train/Avg_Loss_D": avg_loss_d,
            "Train/Avg_Loss_G": avg_loss_g,
            "Train/Avg_Loss_G_Adversarial": avg_loss_g_adv,
            "Train/Avg_Loss_G_L1": avg_loss_g_l1,
        }) # Log against epoch number

        # --- Evaluation Step ---
        if trocr_model is not None: # Only evaluate OCR if model loaded successfully
            avg_val_l1, avg_val_cer, avg_val_wer = evaluate(
                epoch, val_loader, generator, trocr_model, trocr_processor, l1_loss, device
            )
        else:
            # Fallback evaluation if TrOCR failed to load (only L1)
            generator.eval()
            total_l1_loss = 0.0
            with torch.no_grad():
                 for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Evaluating L1 only]"):
                     if batch is None or batch[0] is None: continue
                     damaged, clean, _, _, _ = batch
                     damaged, clean = damaged.to(device), clean.to(device)
                     fake = generator(damaged)
                     total_l1_loss += l1_loss(fake, clean).item()
            avg_val_l1 = total_l1_loss / len(val_loader) if len(val_loader) > 0 else 0
            print(f"Epoch {epoch+1} Validation L1 Loss: {avg_val_l1:.4f} (OCR Eval Skipped)")
            wandb.log({"Validation/L1_Loss": avg_val_l1})
            generator.train()


        # --- Save Sample Images ---
        # Note: Uses the standard save_sample_images which doesn't show OCR text.
        # You could adapt save_sample_images_ocr from the other script if needed.
        if (epoch + 1) % args.save_img_interval == 0 or epoch == args.epochs - 1:
             print("Saving sample images...")
             save_sample_images(epoch + 1, val_loader, generator, device, os.path.join(args.output_dir, "samples"), wandb_log=True)

        # --- Save Checkpoints ---
        # Save latest model
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'val_loss_l1': avg_val_l1, # Save L1 loss
            'args': args
        }, os.path.join(args.output_dir, "checkpoints", "latest_checkpoint.pth"))

        # Save best generator based on validation L1 loss
        if avg_val_l1 < best_val_loss:
            print(f"Validation L1 improved ({best_val_loss:.4f} -> {avg_val_l1:.4f}). Saving best generator...")
            best_val_loss = avg_val_l1
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                 'val_loss_l1': best_val_loss,
                 'args': args
            }, os.path.join(args.output_dir, "checkpoints", "best_generator.pth")) # Save only generator


    print("Training Finished.")
    wandb.finish()

if __name__ == "__main__":
    main()
