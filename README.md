# PTP

## Overview

PTP is a pipeline for processing pSILAC-TMT data and training predictive models. This project includes:

Scripts to preprocess the raw pSILAC-TMT dataset
Model training and evaluation scripts
Utilities for handling experimental and computational pipelines

## Prerequisites

- Python 3.11.4
- Git
- conda

## Setup Steps

1. Clone the repository

   ```bash
   git clone https://github.com/stoneyakoya/PTP
   cd PTP
   ```
2. Create a virtual environment

   ```bash
   conda create -p ./venv python==3.11.4
   ```
3. Activate the virtual environment

   ```bash
   conda activate ./venv
   ```
4. Install PyTorch
   Visit [PyTorch official website](https://pytorch.org/get-started/locally/) and follow the installation command for your system.
5. Install required packages

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Obtain the pSILAC-TMT data from the supplementary materials of the publication by Jana Zecha et al., Nature Communications, 13:165, 2022; (PXD023218).
2. Export the relevant pSILAC-TMT sheet (Supplementary Dataset 1) as a CSV file.
3. Place the CSV file in data/raw. For consistency, we recommend naming it:
   ```bash
   pSILAC_TMT.csv
   ```

Important: Make sure you have permission to use and share this dataset according to the original publication’s terms.

## Preprocessing

Once the data is placed in the data/raw directory, run the preprocessing script:

```bash
   python scripts/preprocess_data.py
```

This script will perform any data cleaning, filtering, or formatting necessary for training. Processed data will be saved in a designated location.

## Train&Test

Simply run the training and testing process with:

```
python src/train_test.py
```

By default, the trained models are saved in data/models.
The results (predictions, metrics, logs, etc.) are saved in data/results.

If you want to adjust parameters (e.g., learning rate, batch size, model architecture, etc.), you can either:

- Modify them directly in the main() function inside src/train_test.py, or
- Use command-line arguments

### Example Usage with Command-Line Arguments

You can customize the training process by passing command-line arguments to `train_test.py`. For example:

`python src/train_test.py --batch_size 8 --lr 0.001 --num_epochs 100 `

In this example:

* `--batch_size 8` sets the batch size to 8.
* `--lr 0.001` sets the learning rate to 0.001.
* `--num_epochs 10` sets the number of training epochs to 10.

### Adjustable Parameters

The following parameters can be modified via command-line arguments. If not specified, the script uses the default values shown below:

| Parameter                  | Description                                                                                                                    | Default Value             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| `--model_type`           | Type of model architecture (e.g., Transformer, LSTM).                                                                          | `"TimeSeq_LSTM"`        |
| `--train_path`           | Path to the training dataset CSV file.                                                                                         | `"data/train.csv"`      |
| `--val_path`             | Path to the validation dataset CSV file.                                                                                       | `"data/validation.csv"` |
| `--test_path`            | Path to the testing dataset CSV file.                                                                                          | `"data/test.csv"`       |
| `--input_cls`            | Whether to use a protein cls embedding input format (True/False).                                                             | `False`                 |
| `--datasize`             | Number of data points to use during training. Use `"all"` to include the entire dataset, or specify an integer for a subset. | "all"                     |
| `--batch_size`           | Number of samples per training batch.                                                                                          | `8`                     |
| `--input_dim`            | Input feature dimension.                                                                                                       | `512`                   |
| `--embed_dim`            | Embedding dimension for the model.                                                                                             | `512`                   |
| `--n_heads`              | Number of attention heads (if using Transformer).                                                                              | `4`                     |
| `--num_layers`           | Number of layers in the model.                                                                                                 | `2`                     |
| `--dropout`              | Dropout rate for regularization.                                                                                               | `0.2`                   |
| `--activation_func`      | Activation function to use (e.g., ReLU, LeakyReLU).                                                                            | `"ReLU"`                |
| `--lr`                   | Learning rate for the optimizer.                                                                                               | `0.001`                 |
| `--num_epochs`           | Number of training epochs.                                                                                                     | `1`                     |
| `--criterion`            | Loss function to use (e.g., MSE, MAE).                                                                                         | `"MSE"`                 |
| `--optimizer`            | Optimizer to use (e.g., Adam, SGD).                                                                                            | `"Adam"`                |
| `--early_stop_patience`  | Number of epochs to wait before stopping if no improvement is observed.                                                        | 6                         |
| `--early_stop_threshold` | Minimum improvement required to reset early stopping.                                                                          | `0.01`                  |
| `--scheduler_patience`   | Number of epochs to wait before adjusting the learning rate.                                                                   | 3                         |
| `--scheduler_threshold`  | Minimum improvement required to adjust the learning rate.                                                                      | `0.02`                  |
| `--model_save_path`      | Path to save the trained model.                                                                                                | `"data/models/...pth"`  |
| `--result_save_dir`      | Directory to save results (predictions, metrics).                                                                              | `"data/results/.../"`   |
| `--plt_save_dir`         | Directory to save training and evaluation plots.                                                                               | `"data/plots/.../"`     |

## Project Directory Structure

```plaintext
.
├── data/             # Datasets (raw and processed data)
├── models/           # Trained model files
├── README.md         # Project overview and setup instructions
├── requirements.txt  # Python dependencies
├── scripts/          # Task-specific scripts (e.g., data preprocessing)
├── src/              # Core source code (e.g., training, evaluation, prediction)
└── venv/             # virtual environment (excluded from version control)
```
# PTP
