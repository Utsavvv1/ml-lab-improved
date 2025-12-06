import json
import os
from train import train_copy_task


def main():
    cfg_path = os.environ.get("ML_LAB_CONFIG", "config.json")
    cfg = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    print(f"Running training with config: {cfg_path}")
    train_copy_task(cfg)


if __name__ == "__main__":
    main()
