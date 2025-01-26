import numpy as np


# Load data file
data_path = "/rds/general/user/ifc24/home/OpenVLA-forget-tune/data/raw-natsgd/data.npy"
fields_path = "/rds/general/user/ifc24/home/OpenVLA-forget-tune/data/raw-natsgd/fields.npy"

# Load the fields (columns metadata)
fields = np.load(fields_path, allow_pickle=True)
print("Fields:", fields)

# Load the actual dataset
data = np.load(data_path, allow_pickle=True)
print("Data shape:", data.shape)
print("Sample data:", data[:2])  # Print the first two rows for inspection
