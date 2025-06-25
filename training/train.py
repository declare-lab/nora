import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from transformers import SchedulerType, get_scheduler
from datasets import RLDSDataset, RLDSBatchTransform
from qwen_vl_utils import process_vision_info
import math
import numpy as np
from tqdm import tqdm
import wandb

logger = get_logger(__name__)

# --- 1. Configuration ---
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 16,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 2,
        num_warmup_steps: int = 1000,
        max_train_steps: int = 100000,
        output_dir: str = '/your_output',
        resume_from_checkpoint: str = '',
        load_model_weights: Optional[str] = None,
        data_root_dir: str = "/your_data_root_dir",
        data_mix: str = "libero_10_no_noops", ## For this, please check out the data mix in /training/datasets/rlds/oxe/mixtures.py
        resize_resolution: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 256_000,
        wandb_project_name: str = "Nora VLA",
        checkpoint_save_frequency: int = 20000,
        logging_frequency: int = 100,
        gradient_clipping: Optional[float] = None, # Add gradient clipping option
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint ## This is used to continue a training by loadinng the optimizer states, model weights etc ... 
        self.load_model_weights = load_model_weights ## This is the path to a pretrained model weights if you want to finetune the model.
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping

# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig, processor: AutoProcessor, is_train: bool = True) -> RLDSDataset:
    """Loads and prepares the RLDS dataset."""
    return RLDSDataset(
        data_root_dir=Path(config.data_root_dir),
        data_mix=config.data_mix,
        batch_transform=RLDSBatchTransform(),
        resize_resolution=config.resize_resolution,
        shuffle_buffer_size=config.shuffle_buffer_size if is_train else None,
        train=is_train,
    )

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on 
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def process_example(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    pixel_values = example['image']
    action = example['action']
    lang = example['lang']
    fast_tokens = fast_tokenizer(action)
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pixel_values},
                {"type": "text", "text": lang},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": vlm_action},
            ],
        },
    ]
    return messages

def collate_fn(examples,processor,fast_tokenizer):
        messages = [process_example(example,fast_tokenizer) for example in examples]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        action_token_min = 151665
        action_token_max = 153712
        labels = batch_input['input_ids'].clone()
        # For each sequence in the batch, find the first occurrence of an action token.
        
        for i in range(labels.size(0)):
            seq = labels[i]
            # Create a mask for tokens within the action token range.
            mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero_indices.numel() > 0:
                first_action_index = nonzero_indices[0].item()
                # Mask out all tokens before the first action token.
                seq[:first_action_index] = -100

            else:
                # If no action token is found, mask the entire sequence.
                seq[:] = -100
        
        labels[labels == processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
        batch_input['labels'] = labels
        return batch_input

# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Loads the model and processor."""
    processor = AutoProcessor.from_pretrained('declare-lab/nora')
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'declare-lab/nora',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" ## Disable flash attention it is not supported in some GPUs
    )
    fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )

    if config.load_model_weights: 
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")

    return model, processor, fast_tokenizer

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    """Main training loop."""
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    accelerator.dataloader_config.dispatch_batches =  False
    logger.info(accelerator.state, main_process_only=False)

    # Initialize Weights and Biases
    if accelerator.is_main_process:
        wandb.init(project=config.wandb_project_name)

    # Load model and processor
    model, processor, fast_tokenizer  = load_model_and_processor(config, accelerator)

    # Load and prepare dataset
    with accelerator.main_process_first():
        train_dataset = load_and_prepare_dataset(config, processor, is_train=True)

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: collate_fn(examples, processor,fast_tokenizer)
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    # Initialize learning rate scheduler
    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=max_train_steps
    )

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Resume from checkpoint if provided
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    # Training loop
    # Right now we assume single node training. I did not test on multi node training.
    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num steps = {config.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    total_loss = 0.0

    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if config.gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                optimizer.step()
                lr_scheduler.step()

            # Logging
            if completed_steps % config.logging_frequency == 0:
                
                
                if accelerator.is_main_process:
                    
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2

                    total_norm = total_norm**0.5
                    lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}")
                    lr = lr_scheduler.get_last_lr()[0]
                    result = {
                        "train_loss": loss.item(),
                        "grad_norm": total_norm,
                        "learning_rate": lr,
                    }
                    wandb.log({"train_loss": loss.item(), "learning_rate": lr}, step=completed_steps)
                

            # Checkpointing
            if completed_steps% config.checkpoint_save_frequency == 0 and completed_steps > 0:
                accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
                if accelerator.is_main_process:
                    
                    summary_data = {"steps": completed_steps, "train_loss": total_loss/config.checkpoint_save_frequency}
                    with open(os.path.join(config.output_dir, "summary.jsonl"), "a") as f:
                        f.write(json.dumps(summary_data) + "\n")
                    logger.info(f"Checkpoint saved at step {completed_steps}")
                    total_loss = 0.0
                    

            
            if completed_steps >= max_train_steps:
                break


    # Save final checkpoint
    accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
    if accelerator.is_main_process:
        
        checkpoint_path = os.path.join(config.output_dir, f"steps_{completed_steps}")
        logger.info(f"Training finished. Final checkpoint saved at {checkpoint_path}")
        wandb.finish()

def main():
    # Initialize training configuration
    config = TrainingConfig()

    # Set up basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Run the training
    train(config)

if __name__ == "__main__":
    main()