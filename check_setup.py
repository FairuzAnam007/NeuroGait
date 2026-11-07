import sys
import numpy as np
import pandas as pd
import sklearn
import torch

print("SETUP CHECK")
print("-------------------------")

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("PyTorch version:", torch.__version__)

print("-------------------------")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")

print("-------------------------")
print("Setup completed successfully.")