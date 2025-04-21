# trocr_utils.py
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoConfig
import torch

def load_trocr_model(model_name: str = "microsoft/trocr-base-stage1", device: torch.device = None):
    """
    Loads the pretrained TrOCR model and processor.

    Args:
        model_name (str): Name of the pretrained TrOCR model to load
                          (e.g., "microsoft/trocr-base-stage1", "microsoft/trocr-base-handwritten").
        device (torch.device, optional): Device to load the model onto. Defaults to None (CPU).

    Returns:
        tuple: (model, processor) where model is VisionEncoderDecoderModel and processor is TrOCRProcessor.
               Returns (None, None) if loading fails.
    """
    print(f"Loading TrOCR model: {model_name}...")
    try:
        # Load processor (handles image preprocessing and tokenization)
        processor = TrOCRProcessor.from_pretrained("/home/hice1/asarswat8/scratch/CS8803-HUM/finetuned_trocr")

        # Load model configuration
        config = AutoConfig.from_pretrained("/home/hice1/asarswat8/scratch/CS8803-HUM/finetuned_trocr")

        # --- Configure model for generation and training ---
        # Set special tokens used for creating decoder_input_ids from labels
        config.decoder_start_token_id = processor.tokenizer.cls_token_id
        config.pad_token_id = processor.tokenizer.pad_token_id
        # Ensure vocab size is set correctly
        config.vocab_size = config.decoder.vocab_size

        # Set beam search parameters (primarily for inference, but good to have)
        config.eos_token_id = processor.tokenizer.sep_token_id
        config.max_length = 128  # Max sequence length for generation
        config.early_stopping = True
        config.no_repeat_ngram_size = 3
        config.length_penalty = 1.0
        config.num_beams = 4

        # Load the model with the modified config
        model = VisionEncoderDecoderModel.from_pretrained("/home/hice1/asarswat8/scratch/CS8803-HUM/finetuned_trocr", config=config)

        if device:
            model.to(device)

        print("TrOCR model and processor loaded successfully.")
        return model, processor

    except OSError as e:
        print(f"Error loading TrOCR model '{model_name}'. Check model name and internet connection.")
        print(f"Error details: {e}")
        # Suggest common models if the default fails
        if "trocr-base-stage1" in model_name:
             print("Try 'microsoft/trocr-base-handwritten' or 'microsoft/trocr-large-stage1'.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading the TrOCR model: {e}")
        return None, None


if __name__ == "__main__":
    # Test loading the model and processor
    print("--- Testing TrOCR Loading ---")
    # Test with default model
    test_model, test_processor = load_trocr_model()
    if test_model and test_processor:
        print(f"Successfully loaded default model: {test_model.config.name_or_path}")
        print(f"Processor vocab size: {test_processor.tokenizer.vocab_size}")
        print(f"Model vocab size: {test_model.config.vocab_size}")

        # Check if special tokens are set
        assert test_model.config.decoder_start_token_id == test_processor.tokenizer.cls_token_id
        assert test_model.config.pad_token_id == test_processor.tokenizer.pad_token_id
        print("Special token IDs configured correctly.")
    else:
        print("Failed to load default TrOCR model.")

    # Test with a different model (optional)
    # test_model_hw, test_processor_hw = load_trocr_model("microsoft/trocr-base-handwritten")
    # if test_model_hw:
    #     print(f"Successfully loaded handwritten model: {test_model_hw.config.name_or_path}")
    # else:
    #     print("Failed to load handwritten TrOCR model.")

    print("--- TrOCR Loading Test Completed ---")

