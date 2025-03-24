import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from jiwer import cer, wer
from dataset.washington_dataset import WashingtonDataset, collate_fn, get_line_ids, get_transcriptions, special_token_map
from model.ocr_model import load_trocr_model
from tqdm import tqdm
import random
import numpy as np

def finetune_trocr():
    """
    Finetune the TrOCR model on the Washington database with evaluation metrics.
    """
    # Define paths (adjust based on your dataset structure)
    transcription_file = 'washingtondb/ground_truth/transcription.txt'
    image_dir = 'washingtondb/data/line_images_normalized'
    sets_dir = 'washingtondb/sets/'
    fold = 1

    # Load transcriptions
    transcriptions = get_transcriptions(transcription_file)

    # Load train and val line IDs
    train_line_ids = get_line_ids(sets_dir, fold, 'train')
    val_line_ids = get_line_ids(sets_dir, fold, 'valid')

    # Load model and processor
    model, processor = load_trocr_model()

    # Create datasets
    train_dataset = WashingtonDataset(train_line_ids, image_dir, transcriptions, special_token_map)
    val_dataset = WashingtonDataset(val_line_ids, image_dir, transcriptions, special_token_map)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=10,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=10,
        num_workers=10,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)

    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training {epoch}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        all_pred_texts = []
        all_label_texts = []
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                outputs = model.generate(pixel_values)
                pred_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                conv = torch.where(batch["labels"] == -100, processor.tokenizer.pad_token_id, batch["labels"])
                label_texts = processor.batch_decode(conv, skip_special_tokens=True)
                all_pred_texts.extend(pred_texts)
                all_label_texts.extend(label_texts)
        avg_cer = cer(all_label_texts, all_pred_texts)
        avg_wer = wer(all_label_texts, all_pred_texts)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")

        num_to_check = 5
        idxs = random.sample(range(len(all_label_texts)), num_to_check)
        for i in idxs:
            label = all_label_texts[i]
            pred = all_pred_texts[i]
            print("-"*50)
            print(f"Label     : '{label}'")
            print(f"Prediction: '{pred}'")
            print("-"*50)


    # Save the finetuned model and processor
    output_dir = "finetuned_trocr"
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Finetuned model and processor saved to {output_dir}")

if __name__ == "__main__":
    # Test the finetuning process
    finetune_trocr()