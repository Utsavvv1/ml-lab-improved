import torch
import glob
import os

def average_checkpoints(pattern="model_epoch_*.pt", output_path="model_averaged.pt", last_n=5):
    # Find all files matching "model_epoch_*.pt"
    checkpoints = sorted(glob.glob(pattern))
    
    if len(checkpoints) == 0:
        print(f"Error: No files found matching {pattern}")
        return
        
    # Take only the last N checkpoints (e.g., last 5)
    to_average = checkpoints[-last_n:]
    print(f"Averaging the following {len(to_average)} checkpoints:")
    for c in to_average:
        print(f" - {c}")
    
    avg_state = None
    
    # Loop through and sum up the weights
    for path in to_average:
        state = torch.load(path)
        if avg_state is None:
            avg_state = state
        else:
            for k in avg_state.keys():
                avg_state[k] += state[k]
                
    # Divide by N to get the average
    for k in avg_state.keys():
        if avg_state[k].is_floating_point():
            avg_state[k] = avg_state[k] / len(to_average)
        else:
            # Handle integer buffers (like batches_tracked in BatchNorm) if any
            avg_state[k] = avg_state[k] // len(to_average)
        
    torch.save(avg_state, output_path)
    print(f"\nSuccess! Averaged model saved to: {output_path}")

if __name__ == "__main__":
    average_checkpoints()