import os
import subprocess
import sys

def run_command(command):
    print(f"--> Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    print("=======================================")
    print("   ML Lab Transformer - Colab Runner   ")
    print("=======================================")
    
    # 1. Install dependencies
    print("\n[Step 1/4] Checking/Installing dependencies...")
    # Colab usually has torch, but we need others
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    # 2. Download Data
    print("\n[Step 2/4] Downloading Multi30k dataset...")
    # Ensure data directory logic uses relative paths correctly
    if not os.path.exists("data/multi30k/train.en"):
        run_command(f"{sys.executable} data/download.py")
    else:
        print("Dataset found in data/multi30k/")

    # 3. Training
    print("\n[Step 3/4] Starting Training...")
    src_train = "data/multi30k/train.en"
    tgt_train = "data/multi30k/train.de"
    
    if not os.path.exists(src_train):
        print(f"Error: Training file {src_train} not found.")
        sys.exit(1)
        
    print("Note: Training will save 'model_real.pt' every epoch.")
    # Pass arguments to train.py
    run_command(f"{sys.executable} train.py {src_train} {tgt_train}")
    
    # 4. Evaluation (Optional, on validation set)
    print("\n[Step 4/4] Evaluating Model...")
    src_val = "data/multi30k/val.en"
    tgt_val = "data/multi30k/val.de"
    
    if os.path.exists("model_real.pt"):
        run_command(f"{sys.executable} evaluate.py model_real.pt {src_val} {tgt_val}")
    else:
        print("model_real.pt not found, skipping evaluation.")

    print("\nAll done! You can download 'model_real.pt' from the file explorer.")

if __name__ == "__main__":
    main()
