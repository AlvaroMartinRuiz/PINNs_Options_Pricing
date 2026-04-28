import torch
import numpy as np

model_path = r'c:\_Alvaro\TFG\TFG_code\results\phase2\pinn_dual_phase2.pt'
checkpoint = torch.load(model_path, map_location='cpu')
history = checkpoint.get('history', {})

k = 'total'
v = history[k]
print(f"{k} values after 15000: {v[15000:]}")

k = 'data'
v = history[k]
print(f"{k} values after 15000: {v[15000:]}")
