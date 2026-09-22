# Simpsons-character-image-classifier

A CNN-based image classifier that identifies Simpsons characters from cropped face/character images, built with PyTorch and trained on Google Colab (T4 GPU).

## What it does

Given an image of a Simpsons character, the model predicts which character it is. It's a multi-class image classification task trained on a labeled dataset of character images, with training designed to handle class imbalance across characters.

## Project structure

```
├── train.ipynb        # Data loading, augmentation, model training, checkpointing
└── inference.ipynb    # Self-contained inference on new/test images
```

## How it works

### Model architecture
A custom CNN (`SimpsonsClassifier`) with 4 convolutional blocks:
- Each block doubles the channel count (64 → 128 → 256 → 512), with two conv layers, batch normalization, ReLU activations, and max pooling per block.
- An adaptive average pool reduces feature maps to a fixed 4×4 size regardless of input resolution.
- The classifier head has two fully connected layers (1024 → 512 units) with dropout (0.5 / 0.5 / 0.3) before the final output layer, sized to the number of character classes.

### Data pipeline (`train.ipynb`)
- Expects `Simpsons.zip` uploaded to Colab, extracted to `/content/archive/characters_train`, with one subfolder per character class containing `.jpg` images.
- Class names and indices are inferred from the folder structure and saved to `class_mapping.json` for reuse without rescanning the dataset.
- An 80/20 train/validation split is stratified by class to preserve class balance.
- Training images are augmented (random resized crop, horizontal flip, rotation, color jitter) to improve generalization; validation images are only resized and normalized.
- A `WeightedRandomSampler` oversamples underrepresented classes during training to counter class imbalance. (Focal loss was also tried for this but is noted in the code as having hurt both F1 and accuracy, so it was dropped in favor of the sampler.)

### Training
- Loss: cross-entropy with label smoothing (0.05).
- Optimizer: Adam, lr `3e-4`, weight decay `1e-4`.
- Scheduler: cosine annealing over up to 50 epochs, decaying to `eta_min=1e-6`.
- Gradient clipping (`max_norm=1.0`) for training stability.
- The checkpoint is saved whenever validation **macro F1** improves (not accuracy), since macro F1 better reflects performance across imbalanced classes.
- Early stopping with patience 7 epochs on validation F1.
- Seeds are fixed throughout (`42`) for reproducibility.
- The best checkpoint (`model.pth`) stores model weights, optimizer/scheduler state, the epoch, validation F1, and both `class_to_idx` and `idx_to_class` mappings — everything needed to run inference without access to the training session.

## Requirements

```
torch
torchvision
pillow
tqdm
scikit-learn
```

Both notebooks are written for Google Colab (`inference.ipynb` uses `google.colab.files` for uploads/downloads; `train.ipynb` uses it to download the trained checkpoint).

## Usage

### Training
1. Open `train.ipynb` in Google Colab with a GPU runtime.
2. Upload `Simpsons.zip` to `/content/` (expected to contain `characters_train/<character_name>/*.jpg`).
3. Run all cells in order. Training runs for up to 50 epochs with early stopping, printing per-epoch loss/accuracy/macro F1 for both train and validation sets.
4. The best checkpoint is saved as `model.pth` and automatically downloaded to your local machine at the end of the run.

### Inference
1. Open `inference.ipynb` in Colab.
2. Update the paths in the final cell:
   - `data_dir` — folder containing the images to classify
   - `model_path` — path to the trained `model.pth` checkpoint
3. Run all cells. The notebook reconstructs the model architecture, loads the checkpoint, walks `data_dir` for `.jpg`/`.jpeg`/`.png` files, and predicts a class for each.
4. Predictions are saved to `results.json` as a mapping of `{filename: predicted_class}`. Images that fail to load or process are recorded as `"unknown"` rather than stopping the run.

## Notes

- `inference.ipynb` redefines the model class independently from `train.ipynb`, so it has no dependency on the training notebook or its Python session — only the `.pth` checkpoint is needed.
- The best model was selected around epoch 51–54; the training notes mention stopping around then because the train/validation F1 gap was widening and validation F1 had plateaued, an early sign of overfitting.

