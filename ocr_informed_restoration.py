# ocr_informed_restoration.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Import F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import argparse
from tqdm import tqdm
import wandb
import random
from jiwer import cer, wer # Import cer, wer directly

# Import models and utilities from other files (Updated Paths)
from model.pix2pix_model import UNetGenerator, PatchDiscriminator
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
# Use the dataset that provides text labels (Updated Path)
from dataset.restoration_dataset import RestorationDatasetWithText

# --- Configuration ---
def parse_args():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="OCR-Informed Pix2Pix GAN Training for Image Restoration")
    # GAN params (Updated defaults/values based on baseline)
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (might need to be smaller due to TrOCR memory)") # Keep smaller for OCR informed
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Learning rate for Generator Adam optimizer")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Learning rate for Discriminator Adam optimizer")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizers")
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="Weight for L1 loss in Generator")
    parser.add_argument("--lambda_ocr", type=float, default=10.0, help="Weight for OCR loss in Generator")
    parser.add_argument("--img_size", type=int, default=512, help="Size to resize images to (must be square)") # Updated default
    # Data params (Updated paths/args based on baseline)
    parser.add_argument("--damaged_dir", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/damaged", help="Directory for damaged images")
    parser.add_argument("--clean_dir", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/data/line_images_normalized", help="Directory for clean (ground truth) images")
    parser.add_argument("--transcription_file", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/ground_truth/transcription.txt", help="Path to the transcription file") # Added
    parser.add_argument("--output_dir", type=str, default="outputs/ocr_informed_imgsize_512_highestOCRLambda", help="Directory to save outputs") # Updated default
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2, help="Number of dataloader workers") # Updated default
    # Logging/Saving params
    parser.add_argument("--log_interval", type=int, default=50, help="Log training stats every N batches")
    parser.add_argument("--save_img_interval", type=int, default=5, help="Save sample images every N epochs")
    parser.add_argument("--wandb_project", type=str, default="HUM_Project", help="Wandb project name") # Updated default
    parser.add_argument("--wandb_run_name", type=str, default="OCR_Informed_512_highestOCRLambda", help="Wadb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)") # Optional
    # TrOCR params
    parser.add_argument("--trocr_model_name", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/finetuned_trocr", help="Name or path of the TrOCR model") # Updated default
    parser.add_argument("--max_text_length", type=int, default=128, help="Max sequence length for TrOCR tokenizer") # Updated default

    return parser.parse_args()

# --- Utility Functions ---
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

def resize_batch(imgs, sizes):
    """ Resizes a batch of images to specified sizes using bilinear interpolation. """
    out = [F.interpolate(img.unsqueeze(0),
                         size=tuple(sz), # Ensure size is a tuple (height, width)
                         mode="bilinear",
                         align_corners=False).squeeze(0)
           for img, sz in zip(imgs, sizes)]
    # Check if output is empty before stacking if needed, but returning list is safer
    return out # Return list of tensors

# Updated save_sample_images_ocr to match baseline's save_sample_images structure
def save_sample_images_ocr(epoch: int, loader: DataLoader, generator: nn.Module, trocr_model: nn.Module, trocr_processor: any,
                           device: torch.device, output_dir: str, wandb_log: bool = True, num_samples: int = 3):
    """ Saves and logs sample images along with OCR predictions, matching baseline structure. """
    generator.eval()
    trocr_model.eval() # Ensure TrOCR is also in eval mode

    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Warning: Validation loader exhausted, cannot save sample images.")
        generator.train()
        return # Exit if loader is exhausted

    if batch is None or batch[0] is None:
        print("Warning: Skipping sample image saving due to corrupted batch.")
        generator.train()
        return

    # Unpack batch data including original sizes
    damaged, clean, text_labels, damaged_fnames, clean_fnames, (original_width, original_height) = batch
    damaged, clean, text_labels = damaged.to(device), clean.to(device), text_labels.to(device)

    with torch.no_grad():
        fake = generator(damaged)

        # --- Select Random Samples ---
        if damaged.size(0) < num_samples:
             print(f"Warning: Batch size ({damaged.size(0)}) is smaller than num_samples ({num_samples}). Using batch size.")
             num_samples = damaged.size(0)

        random_idx = torch.randperm(damaged.size(0))[:num_samples]
        damaged_s = damaged[random_idx]
        clean_s = clean[random_idx]
        fake_s = fake[random_idx]
        text_labels_s = text_labels[random_idx]
        damaged_fnames_s = [damaged_fnames[i] for i in random_idx]
        clean_fnames_s = [clean_fnames[i] for i in random_idx]
        original_width_s = original_width[random_idx]
        original_height_s = original_height[random_idx]
        original_size_s = list(zip(original_height_s.tolist(), original_width_s.tolist())) # Use .tolist() for clean zip

        # --- Denormalize and Resize Images ---
        damaged_s_denorm = denormalize(damaged_s)
        clean_s_denorm = denormalize(clean_s)
        fake_s_denorm = denormalize(fake_s)

        # Resize images back to original size for saving and potentially OCR
        damaged_resized = resize_batch(damaged_s_denorm, original_size_s)
        clean_resized = resize_batch(clean_s_denorm, original_size_s)
        fake_resized = resize_batch(fake_s_denorm, original_size_s) # List of tensors

        # --- Get OCR Predictions on Resized Fake Images ---
        # Prepare resized fake images for TrOCR processor (needs CPU tensors or PIL)
        fake_resized_cpu = [img.cpu() for img in fake_resized]
        try:
            pixel_values = trocr_processor(images=fake_resized_cpu, return_tensors="pt", do_rescale=False).pixel_values.to(device)
            generated_ids = trocr_model.generate(pixel_values, max_length=trocr_processor.tokenizer.model_max_length)
            pred_texts = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"Error during TrOCR generation for sample images: {e}")
            pred_texts = ["OCR Error"] * num_samples


        # Decode ground truth labels
        gt_texts = []
        for label_tensor in text_labels_s:
             label_tensor_cpu = label_tensor.cpu()
             label_tensor_cpu[label_tensor_cpu == -100] = trocr_processor.tokenizer.pad_token_id
             gt_texts.append(trocr_processor.tokenizer.decode(label_tensor_cpu, skip_special_tokens=True))


    # --- Prepare Images for Logging/Saving (Individual Rows) ---
    sample_rows = []
    wandb_images_log = [] # Renamed to avoid conflict
    wandb_table_data = []
    wandb_table_cols = ["ID", "Damaged", "Generated", "Clean", "Predicted Text", "Ground Truth Text"]

    for idx in range(num_samples):
        # Create a grid for each sample: [Damaged | Generated | Clean]
        # Ensure tensors are on CPU for make_grid if necessary
        row_grid = make_grid(
            torch.stack([damaged_resized[idx].cpu(),
                         fake_resized[idx].cpu(),
                         clean_resized[idx].cpu()]),
            nrow=1, # Stack vertically if nrow=1, horizontally if nrow=3
            padding=3,
            normalize=False # Already in [0, 1] range
        )
        sample_rows.append(row_grid)

        # Save individual sample grid
        out_f = os.path.join(output_dir, f"epoch_{epoch:03d}_sample_{idx}.png")
        save_image(row_grid, out_f)

        # Prepare wandb logging
        if wandb_log:
            try:
                # Log grid as wandb.Image
                img_caption = f"Epoch {epoch} - {clean_fnames_s[idx]} :: GT: '{gt_texts[idx]}' :: Pred: '{pred_texts[idx]}'"
                wandb_img = wandb.Image(row_grid, caption=img_caption)
                wandb_images_log.append(wandb_img)

                # Prepare data for table (optional, can be verbose)
                wandb_table_data.append([
                    idx,
                    wandb.Image(damaged_resized[idx].cpu()),
                    wandb.Image(fake_resized[idx].cpu()),
                    wandb.Image(clean_resized[idx].cpu()),
                    pred_texts[idx],
                    gt_texts[idx]
                ])
            except Exception as e:
                 print(f"Warning: Failed to prepare image {idx} for wandb logging: {e}")


    # --- Log to Wandb ---
    if wandb_log and wandb_images_log:
        try:
            wandb.log({"Sample Comparisons": wandb_images_log})
            # Optional: Log table
            ocr_table = wandb.Table(columns=wandb_table_cols, data=wandb_table_data)
            wandb.log({"OCR Sample Table": ocr_table})
        except Exception as e:
            print(f"Warning: Failed to log samples to wandb: {e}")

    generator.train() # Set generator back to training mode

# --- Training Loop ---
def train_one_epoch(epoch: int, dataloader: DataLoader,
                    generator: nn.Module, discriminator: nn.Module,
                    trocr_model: nn.Module, trocr_processor: any, # Add TrOCR components
                    optimizerG: optim.Optimizer, optimizerD: optim.Optimizer,
                    bce_loss: nn.Module, l1_loss: nn.Module,
                    lambda_l1: float, lambda_ocr: float, # Add OCR loss weight
                    device: torch.device, log_interval: int, args: argparse.Namespace):
    """ Trains the OCR-informed GAN for one epoch. """
    generator.train()
    discriminator.train()
    trocr_model.eval() # Keep TrOCR in evaluation mode as it's frozen

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
    total_loss_d = 0.0
    total_loss_g = 0.0
    total_loss_g_adv = 0.0
    total_loss_g_l1 = 0.0
    total_loss_g_ocr = 0.0 # Track OCR loss

    for i, batch in enumerate(pbar):
        if batch is None or batch[0] is None:
             print(f"Warning: Skipping corrupted training batch {i}")
             continue

        # Unpack batch data (includes text labels and original sizes)
        damaged, clean, text_labels, _, _, (original_width, original_height) = batch
        damaged = damaged.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        text_labels = text_labels.to(device, non_blocking=True) # Shape: [batch_size, seq_len]

        # --- Train Discriminator ---
        optimizerD.zero_grad(set_to_none=True) # More efficient zeroing
        with torch.no_grad(): # Generate fake image without tracking gradients for disc
             fake_detached = generator(damaged)
        disc_real_output = discriminator(damaged, clean)
        disc_fake_output = discriminator(damaged, fake_detached)
        real_labels_d = torch.ones_like(disc_real_output, device=device)
        fake_labels_d = torch.zeros_like(disc_fake_output, device=device)
        lossD_real = bce_loss(disc_real_output, real_labels_d)
        lossD_fake = bce_loss(disc_fake_output, fake_labels_d)
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward()
        optimizerD.step()

        # --- Train Generator ---
        optimizerG.zero_grad(set_to_none=True)

        # Generate fake image (keep attached to graph)
        fake = generator(damaged)

        # 1. Adversarial Loss (from Discriminator)
        # We need to run the discriminator again on the attached 'fake'
        disc_fake_output_for_g = discriminator(damaged, fake)
        real_labels_g = torch.ones_like(disc_fake_output_for_g, device=device) # Target is real
        lossG_adv = bce_loss(disc_fake_output_for_g, real_labels_g)

        # 2. L1 Loss (Pixel-wise similarity)
        lossG_l1 = l1_loss(fake, clean)

        # 3. OCR Loss (using frozen TrOCR)
        lossG_ocr = torch.tensor(0.0, device=device) # Initialize to zero
        if lambda_ocr > 0: # Only compute if weight is positive
            try:
                # Denormalize GAN output [-1, 1] -> [0, 1] for processor
                fake_denorm_cpu = denormalize(fake).cpu() # Move to CPU for resizing/processing

                # Resize fake images to original size before OCR processing
                original_size = list(zip(original_height.tolist(), original_width.tolist()))
                fake_resized_cpu = resize_batch(fake_denorm_cpu, original_size) # List of CPU tensors

                # Use the processor to get pixel_values
                pixel_values = trocr_processor(images=fake_resized_cpu, return_tensors="pt", do_rescale=False).pixel_values.to(device)

                # Forward pass through TrOCR model
                outputs = trocr_model(pixel_values=pixel_values, labels=text_labels)
                lossG_ocr = outputs.loss # Extract the loss

                if lossG_ocr is None or not torch.isfinite(lossG_ocr):
                    print(f"Warning: TrOCR model returned invalid loss: {lossG_ocr}. Setting OCR loss to 0 for this batch.")
                    lossG_ocr = torch.tensor(0.0, device=device)

            except Exception as e:
                print(f"\nError during TrOCR forward pass or loss calculation in training: {e}")
                lossG_ocr = torch.tensor(0.0, device=device) # Default to zero loss on error


        # Combined Generator Loss
        lossG = lossG_adv + (lambda_l1 * lossG_l1) + (lambda_ocr * lossG_ocr)

        # Backpropagate and update generator
        lossG.backward()
        optimizerG.step()

        # --- Logging ---
        loss_d_item = lossD.item()
        loss_g_item = lossG.item()
        loss_g_adv_item = lossG_adv.item()
        loss_g_l1_item = lossG_l1.item()
        loss_g_ocr_item = lossG_ocr.item() if isinstance(lossG_ocr, torch.Tensor) else lossG_ocr

        total_loss_d += loss_d_item
        total_loss_g += loss_g_item
        total_loss_g_adv += loss_g_adv_item
        total_loss_g_l1 += loss_g_l1_item
        total_loss_g_ocr += loss_g_ocr_item

        pbar.set_postfix({
            "L_D": f"{loss_d_item:.3f}",
            "L_G": f"{loss_g_item:.3f}",
            "G_Adv": f"{loss_g_adv_item:.3f}",
            "G_L1": f"{loss_g_l1_item:.3f}",
            "G_OCR": f"{loss_g_ocr_item:.3f}"
        })

        # Log to wandb periodically
        if i % log_interval == 0:
            step = epoch * len(dataloader) + i
            wandb.log({
                "Train/Loss_D": loss_d_item,
                "Train/Loss_G": loss_g_item,
                "Train/Loss_G_Adversarial": loss_g_adv_item,
                "Train/Loss_G_L1": loss_g_l1_item,
                "Train/Loss_G_OCR": loss_g_ocr_item,
                "LearningRate/Generator": optimizerG.param_groups[0]['lr'],
                "LearningRate/Discriminator": optimizerD.param_groups[0]['lr'],
            })

    avg_loss_d = total_loss_d / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_g = total_loss_g / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_g_adv = total_loss_g_adv / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_g_l1 = total_loss_g_l1 / len(dataloader) if len(dataloader) > 0 else 0
    avg_loss_g_ocr = total_loss_g_ocr / len(dataloader) if len(dataloader) > 0 else 0


    return avg_loss_d, avg_loss_g, avg_loss_g_adv, avg_loss_g_l1, avg_loss_g_ocr

# --- Evaluation (Adapted from baseline_with_ocr_eval) ---
@torch.no_grad()
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

        # Unpack batch data (includes text labels and original sizes)
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
        fake_denorm_cpu = denormalize(fake).cpu()

        # Resize fake images to original size before OCR processing
        original_size = list(zip(original_height.tolist(), original_width.tolist()))
        fake_resized_cpu = resize_batch(fake_denorm_cpu, original_size) # List of CPU tensors

        try:
            # Prepare input for TrOCR processor
            pixel_values = trocr_processor(images=fake_resized_cpu, return_tensors="pt", do_rescale=False).pixel_values.to(device)

            # Generate text using the TrOCR model
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
        except Exception as e:
            print(f"Error during TrOCR processing/generation in evaluation: {e}")
            # Append placeholders to maintain list lengths if needed for error calculation
            batch_size = damaged.size(0)
            all_pred_texts.extend(["OCR Error"] * batch_size)
            # Attempt to decode GT texts anyway or add GT placeholders
            gt_placeholders = []
            for label_tensor in text_labels:
                 try:
                     label_tensor_cpu = label_tensor.cpu()
                     label_tensor_cpu[label_tensor_cpu == -100] = trocr_processor.tokenizer.pad_token_id
                     decoded = trocr_processor.tokenizer.decode(label_tensor_cpu, skip_special_tokens=True)
                     gt_placeholders.append(decoded)
                 except:
                     gt_placeholders.append("GT Decode Error")
            all_gt_texts.extend(gt_placeholders)


        pbar.set_postfix({"Val_L1": f"{batch_l1:.4f}"})

    # --- Calculate Overall Metrics ---
    avg_l1_loss = total_l1_loss / len(loader) if len(loader) > 0 else 0

    # Calculate CER and WER using jiwer functions
    avg_cer = 1.0 # Default to max error
    avg_wer = 1.0 # Default to max error
    if all_gt_texts and all_pred_texts and len(all_gt_texts) == len(all_pred_texts): # Ensure lists are valid and same length
        try:
            avg_wer = wer(all_gt_texts, all_pred_texts)
            avg_cer = cer(all_gt_texts, all_pred_texts)
            print(f"Epoch {epoch+1} Validation L1 Loss: {avg_l1_loss:.4f}, CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")
        except Exception as e:
            print(f"Error calculating CER/WER with jiwer: {e}")
            # print("Ground truth samples:", all_gt_texts[:5])
            # print("Prediction samples:", all_pred_texts[:5])
    else:
         print(f"Epoch {epoch+1} Validation L1 Loss: {avg_l1_loss:.4f}. CER/WER calculation skipped (data mismatch or empty). GT: {len(all_gt_texts)}, Pred: {len(all_pred_texts)}")


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
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, name=args.wandb_run_name)
    print(f"Wandb run URL: {wandb.run.get_url()}")

    # --- Create Directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    # --- Load TrOCR Model (Frozen) ---
    print("Loading TrOCR model...")
    trocr_model_dir = "/home/hice1/asarswat8/scratch/CS8803-HUM/best_finetuned_trocr"
    trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_dir).to(device)
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_dir)
    if trocr_model is None or trocr_processor is None:
        print("Failed to load TrOCR model. Exiting.")
        return

    # Freeze TrOCR model parameters
    for param in trocr_model.parameters():
        param.requires_grad = False
    trocr_model.eval() # Set to evaluation mode
    print("TrOCR model loaded and frozen.")

    # --- Data Loading ---
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3), # Normalize to [-1, 1] for GAN
    ])

    try:
        # Use the dataset that includes text processing
        full_dataset = RestorationDatasetWithText(
            args.damaged_dir,
            args.clean_dir,
            args.transcription_file, # Pass transcription file path
            transform=img_transform,
            text_processor=trocr_processor,
            max_text_length=args.max_text_length
        )
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return
    except FileNotFoundError as e:
         print(f"Error: Data directory or transcription file not found - {e}.")
         return
    except Exception as e:
        print(f"An unexpected error occurred during dataset initialization: {e}")
        return


    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size <= 0: # Check val_size too
        print(f"Error: Invalid train/validation split sizes ({train_size}/{val_size}) for dataset length {len(full_dataset)}.")
        return
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    print(f"Dataset size: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Initialize GAN Models ---
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)

    # --- Initialize Optimizers and Loss Functions ---
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))

    bce_loss = nn.BCELoss() # Discriminator uses Sigmoid
    l1_loss = nn.L1Loss()

    # Watch GAN models with wandb
    wandb.watch(generator, log="gradients", log_freq=500, idx=0)
    wandb.watch(discriminator, log="gradients", log_freq=500, idx=1)

    # --- Training Loop ---
    print("Starting OCR-Informed Training...")
    best_val_loss = float('inf') # Track best L1 loss

    for epoch in range(args.epochs):
        # --- Training Step ---
        avg_loss_d, avg_loss_g, avg_loss_g_adv, avg_loss_g_l1, avg_loss_g_ocr = train_one_epoch(
            epoch, train_loader, generator, discriminator,
            trocr_model, trocr_processor, # Pass TrOCR components
            optimizerG, optimizerD,
            bce_loss, l1_loss, args.lambda_l1, args.lambda_ocr, # Pass OCR weight
            device, args.log_interval, args
        )

        # Log average epoch losses
        wandb.log({
            "Epoch": epoch + 1,
            "Train/Avg_Loss_D": avg_loss_d,
            "Train/Avg_Loss_G": avg_loss_g,
            "Train/Avg_Loss_G_Adversarial": avg_loss_g_adv,
            "Train/Avg_Loss_G_L1": avg_loss_g_l1,
            "Train/Avg_Loss_G_OCR": avg_loss_g_ocr,
        }) # Log against epoch number

        # --- Evaluation Step ---
        avg_val_l1, avg_val_cer, avg_val_wer = evaluate(
                epoch, val_loader, generator, trocr_model, trocr_processor, l1_loss, device
            )
        # Wandb logging happens inside evaluate function now

        # --- Save Sample Images (with OCR predictions) ---
        if (epoch + 1) % args.save_img_interval == 0 or epoch == args.epochs - 1:
             print("Saving sample images with OCR...")
             save_sample_images_ocr(epoch + 1, val_loader, generator, trocr_model, trocr_processor,
                                    device, os.path.join(args.output_dir, "samples"), wandb_log=True, num_samples=args.batch_size if args.batch_size <=4 else 4) # Limit samples saved

        # --- Save Checkpoints ---
        # Save latest model
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'val_loss_l1': avg_val_l1,
            'val_cer': avg_val_cer, # Save eval metrics
            'val_wer': avg_val_wer,
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
                 'val_cer': avg_val_cer, # Also save CER/WER for best L1 model
                 'val_wer': avg_val_wer,
                 'args': args
            }, os.path.join(args.output_dir, "checkpoints", "best_generator_l1.pth")) # Suffix indicates metric


    print("Training Finished.")
    wandb.finish()

if __name__ == "__main__":
    main()
