# How to Train on Google Colab

This project is optimized to run on Google Colab (Free Tier).

## Quick Start in Colab

1.  **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Create New Notebook**.
3.  **Enable GPU**: 
    *   Go to `Runtime` -> `Change runtime type`.
    *   Select **T4 GPU** (or any available GPU).
    *   Click `Save`.
4.  **Clone & Run**:
    Copy and paste the following code block into a cell and run it:

```python
# 1. Clone the repository
!git clone https://github.com/Utsavvv1/ml-lab-improved.git
%cd ml-lab-improved

# 2. Run the auto-setup script
# This will:
#  - Install requirements
#  - Download the Multi30k dataset
#  - Train the model (this takes ~1-2 hours depending on epochs)
#  - Evaluate and print BLEU score
!python colab_runner.py
```

## Manual Steps (If preferred)

If you want more control:

```python
%cd ml-lab-improved
!pip install -r requirements.txt
!python data/download.py

# Train (Real World Task)
!python train.py data/multi30k/train.en data/multi30k/train.de

# Evaluate
!python evaluate.py model_real.pt data/multi30k/val.en data/multi30k/val.de
```

## Troubleshooting
*   **Out of Memory**: If the GPU runs out of memory, reduce batch size in `train.py` (look for `batch_size=32` inside `train_real_world`).
*   **Drive Saving**: To save the model to your Google Drive to keep it after runtime disconnects:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    !cp model_real.pt /content/drive/MyDrive/model_real.pt
    ```
