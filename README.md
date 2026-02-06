# Restoring Damaged Historical Documents

A computational approach to restoring text in damaged historical documents using OCR-informed Generative Adversarial Networks. Developed as part of CS8803 at Georgia Tech.

## Overview

Historical documents degrade over time through ink fading, staining, paper erosion, and other physical damage. This project builds a pipeline that:

1. **Fine-tunes a TrOCR model** on 18th-century handwriting from the George Washington Papers
2. **Simulates realistic document damage** (erosion, morphological blackhat, ink fade, ink spill, background aging)
3. **Trains a Pix2Pix GAN** to restore damaged images back to their clean form
4. **Introduces an OCR-informed GAN variant** that uses TrOCR recognition loss as an additional training signal

The key contribution is a combined loss function for the generator:

```
L_total = lambda_adv * L_adv + lambda_L1 * L_L1 + lambda_OCR * L_OCR
```

where `L_OCR` is the cross-entropy loss from a frozen fine-tuned TrOCR model, encouraging the GAN to produce restorations that are not only visually similar but also more readable by OCR.

## Architecture

```
Damaged Image --> [Pix2Pix Generator (UNet)] --> Restored Image
                         |                            |
                  [PatchDiscriminator]          [Frozen TrOCR]
                         |                            |
                    L_adv + L_L1                    L_OCR
                         \                          /
                          \--- L_total = weighted --/
```

- **Generator**: UNet with 8 encoder / 7 decoder blocks, skip connections, and dropout
- **Discriminator**: PatchGAN (70x70 receptive field)
- **OCR Model**: TrOCR (Vision Encoder-Decoder) fine-tuned on Washington database handwriting

## Dataset

[George Washington Historical Document Database](https://fki.tic.heia-fr.ch/databases/washington-database):
- 20 pages from the George Washington Papers (18th century)
- ~656 binarized, normalized text line images
- ~4,894 word images
- ~1,500 unique words across ~650 lines
- 4-fold cross-validation splits provided

## Damage Simulation

Five types of synthetic damage are applied to clean document images:

| Type | Description |
|---|---|
| **Erosion** | Dilates text strokes, simulating ink spread |
| **Morphological Blackhat** | Extracts fine details and inverts, simulating stroke degradation |
| **Ink Fade** | Adds white circular blobs over text, simulating faded ink |
| **Ink Spill** | Adds dark blobs over text, simulating ink stains |
| **Background Aging** | Applies sepia-toned parchment coloring with regional variation |

Multiple damage types are combined per image to create realistic degradation.

## Results

| Model | CER | WER |
|---|---|---|
| TrOCR on Original (Clean) Images | 0.0562 | 0.1281 |
| TrOCR on Damaged Images | 0.1272 | 0.2765 |
| TrOCR on GAN-Restored Images | **0.0843** | **0.1858** |
| TrOCR on GAN-Restored Images (OCR-Informed) | 0.0878 | 0.1956 |

The baseline Pix2Pix GAN reduces character error rate by **33.7%** and word error rate by **32.8%** compared to damaged images.

## Project Structure

```
.
├── baseline.py                     # Pix2Pix GAN training with OCR evaluation
├── ocr_informed_restoration.py     # OCR-informed Pix2Pix GAN training
├── finetune_ocr.py                 # Fine-tune TrOCR on Washington database
├── evaluate_images.py              # Evaluate TrOCR on clean vs damaged images
├── evaluate_restoration.py         # Evaluate GAN-restored images via OCR
├── trocr_utils.py                  # TrOCR model loading utilities
├── adversarial_generation/
│   └── adversarial_image_generator.py  # Damage simulation pipeline
├── dataset/
│   └── washington_dataset.py       # Dataset loader for Washington database
├── model/
│   ├── dcgan_model.py              # DCGAN architecture (experimental)
│   ├── diffusion_model.py          # Diffusion-based restoration (experimental)
│   └── ocr_model.py                # TrOCR wrapper
├── GAN_Training/
│   ├── pix2pix_model.py            # Pix2Pix model definitions
│   └── train.py                    # Standalone GAN training loop
└── washingtondb/                   # George Washington dataset
    ├── data/
    │   ├── line_images_normalized/ # 656 text line images
    │   └── word_images_normalized/ # 4894 word images
    ├── ground_truth/
    │   ├── transcription.txt       # Line-level transcriptions
    │   └── word_labels.txt         # Word-level labels
    └── sets/cv1-cv4/               # Cross-validation splits
```

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU recommended (training is compute-intensive)

### Installation

```bash
git clone https://github.com/AryanSarswat/Historical-Document-Restoration.git
cd Historical-Document-Restoration
pip install -r requirements.txt
```

### Usage

**1. Fine-tune TrOCR on the Washington database:**

```bash
python finetune_ocr.py
```

**2. Generate damaged images:**

```bash
python adversarial_generation/adversarial_image_generator.py
```

**3. Train the baseline Pix2Pix GAN:**

```bash
python baseline.py \
  --damaged_dir washingtondb/damaged \
  --clean_dir washingtondb/data/line_images_normalized \
  --transcription_file washingtondb/ground_truth/transcription.txt \
  --trocr_model_name finetuned_trocr \
  --epochs 100
```

**4. Train the OCR-informed GAN:**

```bash
python ocr_informed_restoration.py \
  --damaged_dir washingtondb/damaged \
  --clean_dir washingtondb/data/line_images_normalized \
  --transcription_file washingtondb/ground_truth/transcription.txt \
  --lambda_ocr 10.0 \
  --epochs 100
```

**5. Evaluate restored images:**

```bash
python evaluate_restoration.py \
  --checkpoint_path outputs/checkpoints/best_generator.pth \
  --trocr_model_dir finetuned_trocr
```

> **Note:** Default paths in the scripts point to a Georgia Tech cluster environment. Override them using the command-line arguments shown above when running locally.

## References

- Fischer, A., Keller, A., Frinken, V., & Bunke, H. (2012). Lexicon-Free Handwritten Word Spotting Using Character HMMs. *Pattern Recognition Letters*, 33(7), 934-942.
- Li, M., et al. (2023). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. *AAAI*.
- Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR*.
