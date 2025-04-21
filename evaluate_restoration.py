import torch
import torch.nn as nn
import torch.nn.functional as F # Import F for resize_batch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image # Optional: for saving restored images
from PIL import Image
import os
import argparse
from tqdm import tqdm
import random
from jiwer import cer, wer # Import cer, wer directly

# Import models and utilities from the training script's project structure
from model.pix2pix_model import UNetGenerator # Assuming this path is correct
from dataset.restoration_dataset import RestorationDatasetWithText
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# --- Utility Functions (Copied from training script) ---
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
    # Ensure inputs are tensors
    if isinstance(imgs, list): # Handle list of tensors if necessary
        if not all(isinstance(img, torch.Tensor) for img in imgs):
             raise TypeError("All elements in 'imgs' list must be PyTorch Tensors.")
    elif not isinstance(imgs, torch.Tensor):
         raise TypeError("'imgs' must be a PyTorch Tensor or a list of Tensors.")

    # Ensure sizes is a list of tuples/lists
    if not isinstance(sizes, (list, tuple)) or not all(isinstance(sz, (list, tuple)) and len(sz) == 2 for sz in sizes):
         raise TypeError("'sizes' must be a list of (height, width) tuples/lists.")

    # If input is a single tensor (batch), split it for individual resizing
    if isinstance(imgs, torch.Tensor) and imgs.dim() == 4: # Batch of images
         img_list = list(torch.unbind(imgs, dim=0))
         if len(img_list) != len(sizes):
              raise ValueError(f"Number of images in batch ({len(img_list)}) must match number of sizes ({len(sizes)}).")
    elif isinstance(imgs, list): # Already a list of images (assume single images)
        img_list = imgs
        if len(img_list) != len(sizes):
              raise ValueError(f"Number of images in list ({len(img_list)}) must match number of sizes ({len(sizes)}).")
    else:
        raise ValueError("Input 'imgs' must be a 4D Tensor (batch) or a list of 3D Tensors.")


    out = [F.interpolate(img.unsqueeze(0), # Add batch dim for interpolate
                         size=tuple(sz),    # Ensure size is a tuple (height, width)
                         mode="bilinear",
                         align_corners=False).squeeze(0) # Remove batch dim
           for img, sz in zip(img_list, sizes)]
    # Return list of tensors. Stacking might fail if sizes are different.
    return out

# --- Evaluation Function ---
@torch.no_grad() # Use no_grad for evaluation efficiency
def evaluate_restored(
    generator: nn.Module,
    trocr_model: nn.Module,
    trocr_processor: any,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str = None, # Optional: directory to save some restored images
    num_save_samples: int = 5 # Number of samples to save
):
    """ Evaluates the GAN generator's restored images using TrOCR. """
    generator.eval()
    trocr_model.eval()

    all_pred_texts_restored = []
    all_gt_texts = []
    saved_count = 0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Will save {num_save_samples} sample restored images to {output_dir}")

    pbar = tqdm(dataloader, desc="Evaluating Restored Images")
    for batch in pbar:
        if batch is None or batch[0] is None:
            print(f"Warning: Skipping corrupted validation batch")
            continue

        # Unpack batch data - Assuming dataset yields transformed images now
        # The 'damaged' tensor is already transformed by img_transform
        damaged_transformed, _, text_labels, damaged_fnames, _, (original_width, original_height) = batch

        damaged_transformed = damaged_transformed.to(device, non_blocking=True)
        text_labels = text_labels.to(device) # Shape: [batch_size, seq_len]

        # --- Generate Restored Image using GAN Generator ---
        restored_transformed = generator(damaged_transformed) # Output is [-1, 1]

        # --- Prepare Restored Image for TrOCR ---
        # 1. Denormalize GAN output [-1, 1] -> [0, 1]
        restored_denorm_cpu = denormalize(restored_transformed).cpu() # Move to CPU for resizing

        # 2. Resize restored images back to original size for TrOCR
        original_size_list = list(zip(original_height.tolist(), original_width.tolist()))
        try:
            # Ensure tensors are 3D (C, H, W) before passing to resize_batch if needed
            # resize_batch expects a batch tensor (N, C, H, W) or list of (C, H, W)
            restored_resized_cpu = resize_batch(restored_denorm_cpu, original_size_list) # List of CPU tensors
        except Exception as e:
            print(f"\nError during resize_batch: {e}")
            print(f"Input tensor shape: {restored_denorm_cpu.shape}")
            print(f"Target sizes: {original_size_list}")
            continue # Skip batch on error

        # --- TrOCR Inference on Resized Restored Images ---
        try:
            # 3. Process resized images with TrOCR processor
            # Needs list of PIL Images or list of Tensors/Numpy arrays
            pixel_values = trocr_processor(images=restored_resized_cpu, return_tensors="pt", do_rescale=False).pixel_values.to(device)

            # 4. Generate text using the TrOCR model
            generated_ids = trocr_model.generate(pixel_values, max_length=trocr_processor.tokenizer.model_max_length)

            # 5. Decode generated IDs to text
            batch_pred_texts = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_pred_texts_restored.extend(batch_pred_texts)

            # 6. Decode ground truth labels (handle -100 padding)
            batch_gt_texts = []
            for label_tensor in text_labels:
                label_tensor_cpu = label_tensor.cpu()
                label_tensor_cpu[label_tensor_cpu == -100] = trocr_processor.tokenizer.pad_token_id
                decoded = trocr_processor.tokenizer.decode(label_tensor_cpu, skip_special_tokens=True)
                batch_gt_texts.append(decoded)
            all_gt_texts.extend(batch_gt_texts)

            # Optional: Save some sample outputs
            if output_dir and saved_count < num_save_samples:
                 for i in range(min(len(restored_resized_cpu), num_save_samples - saved_count)):
                     try:
                         fname = os.path.splitext(os.path.basename(damaged_fnames[i]))[0]
                         save_path = os.path.join(output_dir, f"{fname}_restored.png")
                         # restored_resized_cpu is a list of tensors
                         save_image(restored_resized_cpu[i], save_path)
                         saved_count += 1
                     except Exception as save_e:
                         print(f"Warning: Could not save sample image: {save_e}")


        except Exception as e:
            print(f"\nError during TrOCR processing/generation in evaluation: {e}")
            # Append placeholders to maintain list lengths if needed for error calculation
            current_batch_size = damaged_transformed.size(0)
            all_pred_texts_restored.extend(["OCR Error"] * current_batch_size)
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

    # --- Calculate Overall Metrics ---
    avg_cer = 1.0 # Default to max error
    avg_wer = 1.0 # Default to max error
    if all_gt_texts and all_pred_texts_restored and len(all_gt_texts) == len(all_pred_texts_restored):
        try:
            avg_wer = wer(all_gt_texts, all_pred_texts_restored)
            avg_cer = cer(all_gt_texts, all_pred_texts_restored)
            print(f"\nRestored Image Evaluation Results:")
            print(f"  CER: {avg_cer:.4f}")
            print(f"  WER: {avg_wer:.4f}")
        except Exception as e:
            print(f"Error calculating CER/WER with jiwer: {e}")
    else:
         print(f"\nCER/WER calculation skipped (data mismatch or empty). GT: {len(all_gt_texts)}, Pred: {len(all_pred_texts_restored)}")

    # --- Print Sample Comparisons ---
    print("\n--- Sample Restored Predictions ---")
    num_to_check = min(5, len(all_gt_texts))
    if num_to_check > 0:
        idxs = random.sample(range(len(all_gt_texts)), num_to_check)
        for i in idxs:
            label = all_gt_texts[i]
            pred = all_pred_texts_restored[i]
            print("-"*50)
            print(f"Ground Truth: '{label}'")
            print(f"Prediction  : '{pred}'")
            print("-"*50)

    return avg_cer, avg_wer


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR-Informed GAN Restoration")
    # --- Data Args ---
    parser.add_argument("--damaged_dir", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/damaged", help="Directory for damaged images")
    parser.add_argument("--clean_dir", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/data/line_images_normalized", help="Directory for clean (ground truth) images")
    parser.add_argument("--transcription_file", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/ground_truth/transcription.txt", help="Path to the transcription file")
    # --- Model Args ---
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved GAN checkpoint (.pth file)")
    parser.add_argument("--trocr_model_dir", type=str, default="/home/hice1/asarswat8/scratch/CS8803-HUM/best_finetuned_trocr", help="Path to the fine-tuned TrOCR model directory")
    parser.add_argument("--img_size", type=int, default=512, help="Image size the GAN generator was trained with")
    # --- Eval Args ---
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation (adjust based on GPU memory)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Directory to save sample restored images")
    parser.add_argument("--num_save_samples", type=int, default=10, help="Number of sample restored images to save")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load TrOCR Model and Processor ---
    print(f"Loading TrOCR model from {args.trocr_model_dir}...")
    trocr_model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model_dir).to(device)
    trocr_processor = TrOCRProcessor.from_pretrained(args.trocr_model_dir)
    # Configure TrOCR generation parameters (optional but good practice)
    trocr_model.config.max_length = trocr_processor.tokenizer.model_max_length # Use processor's default max length
    trocr_model.config.early_stopping = True
    trocr_model.config.no_repeat_ngram_size = 3
    trocr_model.config.length_penalty = 1.0 # Often 1.0 is fine, adjust if needed
    trocr_model.config.num_beams = 4 # Beam search width
    print("TrOCR model loaded.")

    # --- Load GAN Generator ---
    print(f"Loading GAN generator checkpoint from {args.checkpoint_path}...")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    try:
        # Load checkpoint, ensuring it's mapped to the correct device
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        # Check for keys used in the training script
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
            print("Loaded generator state_dict from checkpoint.")
            # Optionally load args from checkpoint if needed later, e.g., confirm img_size
            # loaded_args = checkpoint.get('args', None)
            # if loaded_args:
            #     print(f"Checkpoint trained with img_size: {loaded_args.img_size}")
            #     # You could potentially override args.img_size here if desired
        else:
             # Attempt to load if the checkpoint *only* contains the state_dict
             generator.load_state_dict(checkpoint)
             print("Loaded generator state_dict directly from checkpoint file.")

    except Exception as e:
        print(f"Error loading generator checkpoint: {e}")
        print("Please ensure the checkpoint path is correct and contains the generator's state_dict.")
        return

    generator.eval() # Set generator to evaluation mode
    print("GAN Generator loaded and set to evaluation mode.")

    # --- Prepare Dataset and DataLoader ---
    # Define the *exact same* transform used during GAN training
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), Image.BICUBIC), # Use BICUBIC like training
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3), # Normalize to [-1, 1]
    ])

    print("Initializing dataset...")
    try:
        eval_dataset = RestorationDatasetWithText(
            args.damaged_dir,
            args.clean_dir,
            args.transcription_file,
            transform=img_transform, # Apply GAN transform
            text_processor=trocr_processor,
            max_text_length=trocr_processor.tokenizer.model_max_length
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    if len(eval_dataset) == 0:
        print("Error: Dataset is empty. Check directories and transcription file.")
        return

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=args.num_workers,
        pin_memory=True # Helps speed up CPU->GPU transfer
    )
    print(f"Dataset loaded with {len(eval_dataset)} samples.")

    # --- Run Evaluation ---
    evaluate_restored(
        generator=generator,
        trocr_model=trocr_model,
        trocr_processor=trocr_processor,
        dataloader=eval_loader,
        device=device,
        output_dir=args.output_dir,
        num_save_samples=args.num_save_samples
    )

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()