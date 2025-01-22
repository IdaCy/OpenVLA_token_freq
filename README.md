# OpenVLA Forgetting and Tokenization Study

## Overview
This repository investigates catastrophic forgetting in the OpenVLA model, with a check on whether the FAST tokenization technique reduces forgetting compared to traditional binning-based tokenization methods.

We analyze how fine-tuning affects the model's latent space, activations, and attention distributions, aiming to provide concrete, quantifiable insights into knowledge retention and degradation.

## Project Structure

```
.
├── analysis/
├── data/
│   ├── raw/
├── scripts/
├── hpc/
├── container/
├── README.md
```

## Objectives

1. **Evaluate catastrophic forgetting:**
   - Investigate if forgetting is observable in the model's latent space.
   - Measure changes in activation and attention patterns pre- and post-fine-tuning.

2. **Compare FAST vs. Naive Binning Tokenization:**
   - Assess if FAST tokenization mitigates forgetting.
   - Compare performance stability between the two approaches.

## Experimental Workflow

### Phase 1: Binning Tokenization
1. Run OpenVLA with the naive binning tokenization on a dataset (e.g., DROID, BridgeV2) and capture intermediate activations and attention values.
2. Run OpenVLA on the same dataset to check for consistency in activation and attention values.
3. Fine-tune the model using binning-based tokenization.
4. Rerun inference and compare pre- and post-fine-tuning activations to detect signs of forgetting.

### Phase 2: FAST Tokenization
1. Run OpenVLA with the FAST tokenization and capture activations and attention values.
2. Fine-tune the model using FAST tokenization.
3. Rerun inference and compare pre- and post-fine-tuning activations and attention values to evaluate stability.

## Evaluation Metrics

- **Activation similarity:**
  - Measure numerical distance using cosine similarity or Mean Squared Error (MSE).
- **Attention divergence:**
  - Compute KL divergence to track changes in attention distributions.
- **Performance drop:**
  - Evaluate action success rates and accuracy changes.
- **Activation stability:**
  - Lower differences should indicate better feature retention.
- **Attention pattern shifts:**
  - Analyze the consistency of attention focus across fine-tuning phases.

## Datasets

- **Conversational Skills Evaluation:**
  - VQA v2 (Image-Question-Answer dataset)
- **Fine-Tuning Dataset:**
  - BridgeData V2 (Robotic demonstration dataset)

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download datasets and place them in `data/raw/`.
3. Run experiments using scripts in `scripts/`, like:
   ```
   python scripts/run_experiment.py --tokenization binning
   python scripts/run_experiment.py --tokenization FAST
   ```

## Running on HPC

```
qsub hpc/train_openvla.pbs
```


## Contact
ida.caspary24[at]imperial.ac.uk

