# PTP

## Prerequisites
- Python 3.11.4
- Git

## Setup Steps

1. Clone the repository
    ```bash
    git clone https://github.com/stoneyakoya/PTP
    cd PTP
    ```

2. Create a virtual environment
    ```bash
    python3 -m venv venv
    ```

3. Activate the virtual environment

    For Unix/MacOS:
    ```bash
    source venv/bin/activate
    ```

    For Windows:
    ```bash
    .\venv\Scripts\activate
    ```
4. Install PyTorch
    Visit [PyTorch official website](https://pytorch.org/get-started/locally/) and follow the installation command for your system.

5. Install required packages
    ```bash
    pip install -r requirements.txt
    ```



## Project Directory Structure

```plaintext
.
├── data/             # Datasets (raw and processed data)
├── models/           # Trained model files
├── notebooks/        # Jupyter notebooks for experiments and analysis
├── README.md         # Project overview and setup instructions
├── requirements.txt  # Python dependencies
├── scripts/          # Task-specific scripts (e.g., data preprocessing)
├── src/              # Core source code (e.g., training, evaluation, prediction)
├── tests/            # Unit and integration tests
├── Untitled-1.py     # Temporary or auxiliary script (consider organizing or removing)
└── venv/             # Python virtual environment (excluded from version control)
# PTP
