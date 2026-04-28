import torch
import os

model_path = r'c:\_Alvaro\TFG\TFG_code\results\phase2\pinn_dual_phase2.pt'
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    history = checkpoint.get('history', {})
    if history:
        for k, v in history.items():
            print(f"{k}: len={len(v)}, last_5={v[-5:]}")
    else:
        print("No history found in checkpoint.")
else:
    print(f"File not found: {model_path}")
