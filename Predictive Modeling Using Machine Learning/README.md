# Malaria Cell Image Classification Using CNN

## Project overview

This project uses a Convolutional Neural Network (CNN) to classify microscopic cell images into two categories:

- **Parasitized** — label `0`
- **Uninfected** — label `1`

Images are loaded from the `cell_images` directory, resized to 150 × 150 pixels, converted to RGB, and normalized to the range 0–1.

## Project files

Keep these files in the `Task_2 ROC Curve` folder:

| File | Description |
|---|---|
| `Model_code.py` | Python source code for loading images, training and evaluating the CNN |
| `malaria_model_10epochs.keras` | Saved trained Keras model |
| `Accuracy_graph.png` | Training and validation accuracy plot |
| `Training and validation loss.png` | Training and validation loss plot |
| `ROC_graph.png` | ROC curve |
| `Confusion_matrix.png` | Confusion matrix at the default 0.5 threshold |
| `Confusion_matrix(Optimal threshold).png` | Confusion matrix at the exploratory threshold selected from the ROC curve |
| `cell_images/` | Dataset folder containing `Parasitized/` and `Uninfected/` subfolders |
| `requirements.txt` | Python package dependencies |

## Dataset source and licence

The dataset is from "https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html"

## CNN architecture

The model uses the following architecture:

1. Convolutional layers with 32, 32 and 64 filters
2. ReLU activations and max-pooling
3. Flatten layer
4. Dense layer with 64 units
5. Dropout layer with a rate of 0.5
6. One sigmoid output unit for binary classification

The model is compiled with binary cross-entropy loss and the RMSprop optimizer.

## Training setup

- Image size: 150 × 150 pixels
- Colour channels: RGB
- Batch size: 64
- Epochs: 10
- Train/test split: 80% / 20%
- Validation split: 20% of the training portion
- Random state: 0
- Stratified train/test split

This gives approximately 64% of the full dataset for training, 16% for validation, and 20% for testing.

## Results

The recorded run produced these results:

| Metric | Result |
|---|---:|
| Test accuracy | 95.70% |
| Test loss | 0.1411 |
| ROC-AUC | 0.9883 |
| Training accuracy (epoch 10) | 96.54% |
| Validation accuracy (epoch 10) | 95.62% |

### Confusion matrix at threshold 0.5

Rows represent actual classes and columns represent predicted classes. Class order: `Parasitized`, then `Uninfected`.

| Actual / Predicted | Parasitized | Uninfected |
|---|---:|---:|
| Parasitized | 2594 | 162 |
| Uninfected | 75 | 2681 |

### Classification report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Parasitized | 0.97 | 0.94 | 0.96 |
| Uninfected | 0.94 | 0.97 | 0.96 |

The ROC-AUC of 0.9883 indicates strong separation between the two classes on the evaluated test data. AUC measures ranking performance across thresholds; it is not the same as accuracy.

**Threshold note:** The exploratory threshold `0.6086665` was selected by maximizing Youden's J statistic using the test-set ROC results. Since the test set was used to choose this threshold, the resulting optimal-threshold confusion matrix is not an independent final test evaluation. For a rigorous evaluation, select a threshold using validation data and evaluate it once on an untouched test set.

## How to run

1. Install the packages listed in `requirements.txt`.
2. Ensure the dataset has this directory structure:

   ```text
   cell_images/
   ├── Parasitized/
   └── Uninfected/
   ```

3. Open `Model_code.py`.
4. Check that the dataset path in the script points to the `cell_images` directory on your computer. Update the path if necessary.
5. Run the script using Python or Spyder.

Running the training script trains the CNN again. To use the existing trained model without retraining, load `malaria_model_10epochs.keras` with TensorFlow Keras.

## Dependencies

See `requirements.txt` for the Python packages used by the project.

