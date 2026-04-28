import torch
import numpy as np

model_path = r'c:\_Alvaro\TFG\TFG_code\results\phase2\pinn_dual_phase2.pt'
checkpoint = torch.load(model_path, map_location='cpu')
history = checkpoint.get('history', {})

for k, v in history.items():
    v_np = np.array(v)
    nans = np.isnan(v_np).sum()
    infs = np.isinf(v_np).sum()
    max_val = np.max(v_np)
    max_idx = np.argmax(v_np)
    print(f"{k}: nans={nans}, infs={infs}, max={max_val:.4e} at idx {max_idx}, len={len(v)}")
    if len(v) > 15000:
        print(f"  Values after 15000: {v[15000:]}")
