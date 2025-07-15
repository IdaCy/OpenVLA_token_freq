#!/usr/bin/env python3
"""
Minimal test script to verify that the custom dataset key ("somethv2_rlds") is recognized.
This script instantiates an RLDSDataset and prints its statistics.
It does not perform any training.
"""

from pathlib import Path
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

# Print available dataset keys to verify our custom key is present.
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
print("Available dataset keys in OXE_DATASET_CONFIGS:")
print(list(OXE_DATASET_CONFIGS.keys()))

# Set the dataset name and data root directory.
# If you have added "somethv2_rlds" in the configs, it should appear in the list above.
dataset_name = "somethv2_rlds"
data_root_dir = Path("data")  # Assumes your data folder is "data/somethv2_rlds/"

# Load a processor to obtain a tokenizer and image processor.
# (This example uses the OpenVLA model as a reference.)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)

# Set up a minimal batch transform.
# The prompt builder here is chosen as PurePromptBuilder.
batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder,
)

# Try to instantiate the dataset.
try:
    dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=(224, 224),  # Adjust based on your expected image size
        shuffle_buffer_size=100_000,
        image_aug=False,  # Set to False for simplicity
    )
    print("\nDataset instantiated successfully.")
    print("Dataset statistics:")
    print(dataset.dataset_statistics)
except Exception as e:
    print("\nFailed to instantiate RLDSDataset:")
    print(e)
