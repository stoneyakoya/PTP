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
This script will perform any data cleaning, filtering, or formatting necessary for training. Processed data will be saved in a designated location (e.g., data/processed).


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
