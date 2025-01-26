import os

# Download requirements file first
requirements_url = "https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt"
requirements_file = "requirements-min.txt"

# Download the requirements file if it doesn't already exist
if not os.path.exists(requirements_file):
    os.system(f"wget {requirements_url} -O {requirements_file}")

# Install dependencies
os.system(f"pip install -r {requirements_file}")
