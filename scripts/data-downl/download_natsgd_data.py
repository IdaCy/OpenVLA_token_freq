import gdown
import os

# Configuration
GOOGLE_DRIVE_FILE_ID = '18zGu4-cnwfqIekRLvv8WlzU69J1CxYWG'
DESTINATION_PATH = 'data/raw-natsgd.zip'


def download_natsgd():
    """Download the NatSGD dataset from Google Drive."""
    if os.path.exists(DESTINATION_PATH):
        print("Dataset already exists. Skipping download.")
        return

    os.makedirs(os.path.dirname(DESTINATION_PATH), exist_ok=True)
    url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
    gdown.download(url, DESTINATION_PATH, quiet=False)
    print("Download complete.")


if __name__ == "__main__":
    download_natsgd()
