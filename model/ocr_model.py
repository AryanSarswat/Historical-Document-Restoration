from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def load_trocr_model(model_name="microsoft/trocr-base-stage1"):
    """
    Load the pretrained TrOCR model and processor.
    
    Args:
        model_name (str): Name of the pretrained model to load (default: "microsoft/trocr-base-handwritten").
    
    Returns:
        tuple: (model, processor) where model is VisionEncoderDecoderModel and processor is TrOCRProcessor.
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 4

    return model, processor

if __name__ == "__main__":
    # Test loading the model and processor
    model, processor = load_trocr_model()



    print("Model test - Model and processor loaded successfully.")