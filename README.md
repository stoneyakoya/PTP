# PTP

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

## Create Datasets
1. Clone the repository
   Please save the pSILAC-TMT sheet from the original supplementary data file supplymentary dataset 1（Jana Zecha et
   al., Nature Communications, 13:165, 2022; PXD023218） as a CSV file and store it in the data/raw directory.
   It is recommended to name the file as pSILAC_TMT.csv.
2. Pre-process
   ```bash
   python scripts/preprocess_data.py
   ```
   
## Train&Test

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
