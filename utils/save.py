import torch
import os


def save_model_state(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_state(model, path, map_location=None):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model
