"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

# Define layers to track
tracked_layers = [0, 4, 9, 14, 19, 24, 29, 31]
tracking_dir = "finetunes/tracking"
os.makedirs(tracking_dir, exist_ok=True)

# Stores activations & attention dynamically
activation_store = {}
attention_store = {}

def activation_hook(layer_name):
    def hook(module, input, output):
        activation_store[layer_name] = output.detach().cpu()
    return hook

def attention_hook(layer_name):
    def hook(module, input, output):
        attention_store[layer_name] = output.detach().cpu()
    return hook

def register_hooks(model):
    # First, try to see if the model has its transformer layers under "language_model.model.layers"
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        base_layers = model.language_model.model.layers
    # Else try "encoder.layers"
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        base_layers = model.encoder.layers
    else:
        raise ValueError("ERROR: Could not locate transformer layers in the model. Check model structure!")
    
    for layer_idx in tracked_layers:
        transformer_layer = base_layers[layer_idx]
        transformer_layer.register_forward_hook(activation_hook(f"layer_{layer_idx}"))
        if hasattr(transformer_layer, "self_attn"):
            transformer_layer.self_attn.register_forward_hook(attention_hook(f"layer_{layer_idx}"))
        elif hasattr(transformer_layer, "attn"):
            transformer_layer.attn.register_forward_hook(attention_hook(f"layer_{layer_idx}"))

def save_tensor(data, filename):
    """Helper function to efficiently save tensors with compression."""
    save_path = os.path.join(tracking_dir, filename)
    if isinstance(data, dict):
        data = {k: v.cpu().half() if torch.is_tensor(v) and v.dtype != torch.float32 else v.cpu() for k, v in data.items()}
    elif torch.is_tensor(data):
        data = data.cpu().half() if data.dtype != torch.float32 else data.cpu()
    torch.save(data, save_path, _use_new_zipfile_serialization=False)

def capture_activations(step):
    save_tensor(activation_store, f"activations_step_{step}.pt")

def capture_attention_weights(step):
    save_tensor(attention_store, f"attention_weights_step_{step}.pt")

def capture_weight_changes(model, step):
    lora_weights = {name: param.detach().cpu() for name, param in model.named_parameters() if "lora" in name and param.requires_grad}
    save_tensor(lora_weights, f"lora_weights_step_{step}.pt")

def capture_gradient_magnitudes(model, step):
    grad_norms = {name: torch.norm(param.grad).item() for name, param in model.named_parameters() if param.grad is not None}
    save_tensor(grad_norms, f"gradient_magnitudes_step_{step}.pt")

def capture_logits(output, step):
    logits = output.logits.detach().cpu()
    save_tensor(logits, f"logits_step_{step}.pt")

def capture_loss(loss, step):
    loss_value = loss.item()
    save_tensor(loss_value, f"loss_step_{step}.pt")

def capture_feature_representations(model, step):
    feature_representations = {layer: activation_store.get(f"layer_{layer}", torch.zeros(1)) for layer in tracked_layers}
    save_tensor(feature_representations, f"feature_representations_step_{step}.pt")

def capture_gradients(model, step):
    gradients = {name: param.grad.detach().cpu() for name, param in model.named_parameters() if param.grad is not None}
    save_tensor(gradients, f"gradients_step_{step}.pt")

def compute_hessian_metric(model, loss):
    loss = loss.mean()
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        grads = torch.autograd.grad(loss, params, create_graph=True)
        hessian_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
    except RuntimeError:
        print("WARNING: Hessian computation failed (probably due to vanishing gradients). Skipping this step.")
        hessian_norm = torch.tensor(-1.0)
    return hessian_norm.detach().cpu()

def capture_hessian_based_metrics(model, loss, step):
    sharpness = compute_hessian_metric(model, loss)
    save_tensor(sharpness, f"hessian_step_{step}.pt")

def capture_sparsity_changes(model, step):
    sparsity = {layer: (activation_store.get(f"layer_{layer}", torch.zeros(1)) != 0).sum().item() for layer in tracked_layers}
    save_tensor(sparsity, f"sparsity_step_{step}.pt")

@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("data")
    dataset_name: str = "somethv2_rlds"
    run_root_dir: Path = Path("finetunes")
    adapter_tmp_dir: Path = Path("adapter-tmp")
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000
    save_latest_checkpoint_only: bool = True
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    wandb_project: str = "openvla"
    wandb_entity: str = "stanford-voltron"
    run_id_note: Optional[str] = None

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    exp_id = (f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
              f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
              f"+lr-{cfg.learning_rate}")
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print("\n==== Model Structure ====\n")
    for name, module in vla.named_modules():
        print(name)
    print("\n=========================\n")

    # Register hooks only on the main process to avoid errors in distributed workers.
    if distributed_state.is_main_process:
        register_hooks(vla)

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                if batch_idx % 200 == 0:
                    capture_activations(vla, batch_idx)
                    capture_attention_weights(vla, batch_idx)
                    capture_logits(output, batch_idx)
                    capture_feature_representations(vla, batch_idx)
                loss = output.loss
                if batch_idx % 1000 == 0:
                    capture_loss(loss, batch_idx)
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            if batch_idx % 200 == 0:
                capture_gradient_magnitudes(vla, batch_idx)
                capture_gradients(vla, batch_idx)
            if batch_idx % 1000 == 0:
                capture_sparsity_changes(vla, batch_idx)
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                if batch_idx % 500 == 0:
                    capture_weight_changes(vla, batch_idx)
                if batch_idx % 1000 == 0:
                    capture_hessian_based_metrics(vla, loss.mean(), batch_idx)
                optimizer.zero_grad()
                progress.update()
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)
                dist.barrier()
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            merged_vla.save_pretrained(run_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                dist.barrier()
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

if __name__ == "__main__":
    finetune()
