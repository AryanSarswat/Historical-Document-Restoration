import os
from PIL import Image
import torch
from torch.utils.data import Dataset

# Mapping for special tokens to standard characters
special_token_map = {
    "s_pt": ".",
    "s_cm": ",",
    "s_qm": "?",
    "s_em": "!",
    "s_s": "s",
    "s_0": "0",
    "s_1": "1",
    "s_2": "2",
    "s_3": "3",
    "s_4": "4",
    "s_5": "5",
    "s_6": "6",
    "s_7": "7",
    "s_8": "8",
    "s_9": "9",
    # Add more mappings as needed based on the dataset
}

def get_transcriptions(transcription_file):
    """
    Load transcriptions from the transcription file.
    
    Args:
        transcription_file (str): Path to the transcription file.
    
    Returns:
        dict: Dictionary mapping line IDs to their raw transcriptions.
    """
    transcriptions = {}
    with open(transcription_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                line_id, transcription = parts
                transcriptions[line_id] = transcription
    return transcriptions

def get_line_ids(sets_dir, fold, split):
    """
    Load line IDs for a specific split and fold.
    
    Args:
        sets_dir (str): Directory containing the split files.
        fold (int): Fold number (e.g., 1, 2, 3, 4).
        split (str): Split type ('train', 'val', 'test').
    
    Returns:
        list: List of line IDs for the specified split and fold.
    """
    file_name = os.path.join(sets_dir, f'cv{fold}/{split}.txt')
    with open(file_name, 'r') as f:
        line_ids = [line.strip() for line in f]
    return line_ids

def transcription_to_text(transcription, special_token_map):
    """
    Convert raw transcription to plain text by handling special tokens.
    
    Args:
        transcription (str): Raw transcription (e.g., "H-e-l-l-o|s_pt").
        special_token_map (dict): Mapping of special tokens to standard characters.
    
    Returns:
        str: Plain text (e.g., "Hello .").
    """
    words = transcription.split('|')
    processed_words = []
    for word in words:
        tokens = word.split('-')
        processed_word = []
        for char in tokens:
            if char in special_token_map:
                processed_word.append(special_token_map[char])
            else:
                processed_word.append(char)
        processed_words.append("".join(processed_word))
    text = ' '.join(processed_words)
    return text

class WashingtonDataset(Dataset):
    def __init__(self, line_ids, image_dir, transcriptions, special_token_map):
        """
        Custom dataset for the Washington database.
        
        Args:
            line_ids (list): List of line IDs.
            image_dir (str): Directory containing the line images.
            transcriptions (dict): Dictionary mapping line IDs to raw transcriptions.
            special_token_map (dict): Mapping for special tokens.
        """
        self.line_ids = line_ids
        self.image_dir = image_dir
        self.transcriptions = transcriptions
        self.special_token_map = special_token_map

        # Process
        raw_transcriptions = [self.transcriptions[line_id] for line_id in self.line_ids]
        self.text = [transcription_to_text(transcription, special_token_map) for transcription in raw_transcriptions]


    def __len__(self):
        return len(self.line_ids)

    def __getitem__(self, index):
        """
        Get a single sample (image and transcription).
        
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: (image, transcription) where image is a PIL Image and transcription is plain text.
        """
        line_id = self.line_ids[index]
        image_path = os.path.join(self.image_dir, f"{line_id}.png")
        image = Image.open(image_path).convert("RGB")  # TrOCR expects RGB images
        transcription = self.text[index]
        return image, transcription

def collate_fn(batch, processor):
    """
    Process a batch of images and transcriptions for TrOCR.
    
    Args:
        batch (list): List of (image, transcription) tuples.
        processor (TrOCRProcessor): Processor to handle image and text processing.
    
    Returns:
        dict: Processed batch with 'pixel_values' and 'labels'.
    """
    images, transcriptions = zip(*batch)
    # Process images (no padding)
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    # Process text (with padding)
    labels = processor.tokenizer(transcriptions, padding="max_length", return_tensors="pt").input_ids
    labels = torch.where(labels == processor.tokenizer.pad_token_id, -100, labels)
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    # Test the dataset
    transcription_file = 'washingtondb/ground_truth/transcription.txt'
    image_dir = 'washingtondb/data/line_images_normalized'
    sets_dir = 'washingtondb/sets/'
    fold = 1
    if os.path.exists(transcription_file) and os.path.exists(image_dir) and os.path.exists(sets_dir):
        train_line_ids = get_line_ids(sets_dir, fold, 'train')
        transcriptions = get_transcriptions(transcription_file)
        dataset = WashingtonDataset(train_line_ids, image_dir, transcriptions, special_token_map)
        image, transcription = dataset[0]
        print(f"Dataset test - Sample image size: {image.size}, transcription: '{transcription}'")
    else:
        print("Dataset test skipped: Required data directories/files not found.")