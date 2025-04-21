from dataset.restoration_dataset import RestorationDatasetWithText
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from jiwer import cer, wer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

def main(trocr_model, trocr_processor):
    # --- Example Usage (requires dummy data and transformers) ---
    print("--- Testing RestorationDatasetWithText ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create dummy data directories
    damaged_dir = "/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/damaged"
    clean_dir = "/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/data/line_images_normalized"
    transcription_file = "/home/hice1/asarswat8/scratch/CS8803-HUM/washingtondb/ground_truth/transcription.txt"
    
    dataset_with_text_proc = RestorationDatasetWithText(
        damaged_dir, 
        clean_dir, 
        transcription_file,
        transform=None,
        text_processor=trocr_processor,
        max_text_length=128 # Small max length for testing
    )

    trocr_model.eval()
    all_pred_texts = []
    all_label_texts = []

    all_pred_text_damaged = []
    all_pred_label_damaged = []

    counter = 0
    with torch.inference_mode():
        for batch in tqdm(dataset_with_text_proc):
            damaged, clean, text_labels, _, _, (original_width, original_height) = batch

            damaged_pixel = trocr_processor(images=[damaged], return_tensors="pt").pixel_values.to(device)
            clean_pixel = trocr_processor(images=[clean], return_tensors="pt").pixel_values.to(device)

            outputs_damaged = trocr_model.generate(damaged_pixel).cpu()
            outputs_clean =  trocr_model.generate(clean_pixel).cpu()

            pred_texts_damaged = trocr_processor.decode(outputs_damaged[0], skip_special_tokens=True)
            pred_texts_clean = trocr_processor.decode(outputs_clean[0], skip_special_tokens=True)

            conv = torch.where(text_labels == -100, trocr_processor.tokenizer.pad_token_id, text_labels)
            label_texts = trocr_processor.decode(conv, skip_special_tokens=True)

            all_pred_texts.append(pred_texts_clean)
            all_label_texts.append(label_texts)
            
            all_pred_text_damaged.append(pred_texts_damaged)
            all_pred_label_damaged.append(label_texts)
            # counter += 1

            # if counter == 100:
            #     break
    
    avg_cer_clean = cer(all_label_texts, all_pred_texts)
    avg_wer_clean = wer(all_label_texts, all_pred_texts)
    print(f"Clean Validation CER: {avg_cer_clean:.4f}, Clean WER: {avg_wer_clean:.4f}")

    avg_cer_damaged = cer(all_pred_label_damaged, all_pred_text_damaged)
    avg_wer_damaged = wer(all_pred_label_damaged, all_pred_text_damaged)
    print(f"Damaged Validation CER: {avg_cer_damaged:.4f}, Damaged WER: {avg_wer_damaged:.4f}")
    
    print("Clean Samples")
    num_to_check = 5
    idxs = random.sample(range(len(all_label_texts)), num_to_check)
    for i in idxs:
        label = all_label_texts[i]
        pred = all_pred_texts[i]
        print("-"*50)
        print(f"Label     : '{label}'")
        print(f"Prediction: '{pred}'")
        print("-"*50)

    print("Damaged Samples")
    num_to_check = 5
    idxs = random.sample(range(len(all_label_texts)), num_to_check)
    for i in idxs:
        label = all_pred_label_damaged[i]
        pred = all_pred_text_damaged[i]
        print("-"*50)
        print(f"Label     : '{label}'")
        print(f"Prediction: '{pred}'")
        print("-"*50)

if __name__ == "__main__":
    # Load a processor (replace with your actual model if needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model_dir = "/home/hice1/asarswat8/scratch/CS8803-HUM/best_finetuned_trocr/"
    trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_dir).to(device)
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_dir)

    trocr_model.config.max_length = 128
    trocr_model.config.early_stopping = True
    trocr_model.config.no_repeat_ngram_size = 3
    trocr_model.config.length_penalty = 1.0
    trocr_model.config.num_beams = 4

    main(trocr_model, trocr_processor)
