

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
#from lerobot_normalization import Normalize, PolicyFeature,NormalizationMode
from lerobot.configs.types import  NormalizationMode, PolicyFeature
from lerobot.policies.normalize import (
    Normalize,
    Unnormalize,
)
import safetensors
from qwen_vl_utils import process_vision_info
import math
import numpy as np
from tqdm import tqdm


import torchvision



logger = get_logger(__name__)

# --- 1. Configuration ---
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 16,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 1,
        num_warmup_steps: int = 1000,
        max_train_steps: int = 60000,
        output_dir: str = './nora_finetune_spatial_mapped',
        resume_from_checkpoint: str = './nora_finetune_spatial_mapped/steps_40000',
        load_model_weights: Optional[str] = None,
        lerobot_dataset_repo_id: str = "lerobot/libero_spatial_image",
        wandb_project_name: str = "Nora VLA with LeRobotDataset",
        checkpoint_save_frequency: int = 20000,
        logging_frequency: int = 100,
        gradient_clipping: Optional[float] = None,
        invert_grippler_action: bool = True,
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.load_model_weights = load_model_weights
        self.lerobot_dataset_repo_id = lerobot_dataset_repo_id
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping
        ## In Nora's pretraining, the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open. While some environments have -1 = open, +1 = close. Setting this to True will invert the gripper action(map -1 to 1, +1 to 0)
        self.invert_grippler_action = invert_grippler_action 
        self.image_key = 'observation.images.image'
        self.action_key = 'action'
        self.task_key = 'task'

# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig) -> LeRobotDataset:
    """Loads and prepares the LeRobot dataset."""
    return LeRobotDataset(config.lerobot_dataset_repo_id)

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format."""
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def inverse_transform_gripper_action(action, binarized_input=True):
    """
    
    Maps the gripper action  to the range [0, 1].
    Args:
        action (torch.Tensor): The action vector with the gripper action as the last dimension,
                             which has been transformed by invert_gripper_action and then
                             normalize_gripper_action.
        binarized_input (bool): Whether the input to normalize_gripper_action was binarized.
                                This affects the inverse transformation.

    """

    action[..., -1] = action[..., -1] * -1.0

    if binarized_input:
        # If the input was binarized, the values are -1 or +1.
        # Just map -1 to 0 and +1 to 1. Note that the previous line we have already flipped the sign.
        action[..., -1] = torch.where(action[..., -1] == -1, 0.0, 1.0)
    else:
        # If not binarized, the inverse of y = 2x - 1 is x = (y + 1) / 2
        action[..., -1] = (action[..., -1] + 1) / 2

    return action


def process_example(example: Dict[str, Any],
                    fast_tokenizer: AutoProcessor,
                    normalizer: Normalize,
                    cfg,
                    ) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    example = normalizer(example)
    pixel_values = example[cfg.image_key]
    pixel_values = torchvision.transforms.functional.to_pil_image(pixel_values)

    if cfg.invert_grippler_action:

        #example[cfg.action_key][...,-1] = torch.where(example[cfg.action_key][-1] == -1, 1.0, 0.0)
        example[cfg.action_key] = inverse_transform_gripper_action(example[cfg.action_key].clone(),binarized_input=True)  
    

    action = example[cfg.action_key].unsqueeze(0)
    lang = example[cfg.task_key]
    fast_tokens = fast_tokenizer(action)
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pixel_values,
                 "resized_height": 224,
                "resized_width": 224,},
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

def collate_fn(examples, processor, fast_tokenizer, normalizer,config):
    messages = [process_example(example, fast_tokenizer, normalizer,config) for example in examples]

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

    for i in range(labels.size(0)):
        seq = labels[i]
        mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
        nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
        if nonzero_indices.numel() > 0:
            first_action_index = nonzero_indices[0].item()
            seq[:first_action_index] = -100
        else:
            seq[:] = -100
    
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch_input['labels'] = labels
    return batch_input

# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoProcessor]:
    """Loads the model and processor."""
    processor = AutoProcessor.from_pretrained('declare-lab/nora')
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'declare-lab/nora',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
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
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps,log_with="wandb")
    accelerator.dataloader_config.dispatch_batches = False
    logger.info(accelerator.state, main_process_only=False)

    
    accelerator.init_trackers(config.wandb_project_name, config=config)
        #wandb.init(project=config.wandb_project_name)

    model, processor, fast_tokenizer = load_model_and_processor(config, accelerator)

    with accelerator.main_process_first():
        dataset = load_and_prepare_dataset(config)
        metadata = LeRobotDatasetMetadata(config.lerobot_dataset_repo_id)
        stats = metadata.stats
       
        if stats['action']['min'][-1]>=0 and config.invert_grippler_action:
            logger.warning("The dataset's action stats indicate that the gripper action is already in the range [0, 1].  You are training with invert_grippler_action = True. Inverting gripper action may not be necessary.")

        features = {
                    'action': PolicyFeature(shape=stats['action']['mean'].shape, type='action'),
                }
        norm_map = {
            'action': NormalizationMode.MIN_MAX,
        }
        normalizer = Normalize(features, norm_map, stats)


    
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: collate_fn(examples, processor, fast_tokenizer, normalizer,config)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps*accelerator.num_processes,
        num_training_steps=config.max_train_steps*accelerator.num_processes
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,lr_scheduler
    )

    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num steps = {config.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    progress_bar = tqdm(range(completed_steps,max_train_steps), disable=not accelerator.is_local_main_process)
    

    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)

                
                    

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if config.gradient_clipping is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                    completed_steps += 1

                    optimizer.step()
                    lr_scheduler.step()

                    if completed_steps % config.logging_frequency == 0:
                        
                        if accelerator.is_main_process:
                            total_norm = 0.0
                            for p in model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2

                            total_norm = total_norm**0.5
                            lr = lr_scheduler.get_last_lr()[0]
                            
                            logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}", main_process_only=True)
                            accelerator.log({"train_loss": loss.item(), "learning_rate": lr,"grad_norm":total_norm}, step=completed_steps)
                    #logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}", main_process_only=True)
                    #accelerator.log({"train_loss": loss.item(), "learning_rate": lr,"grad_norm":total_norm}, step=completed_steps)

            if completed_steps % config.checkpoint_save_frequency == 0 and completed_steps > 0:
                accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
                

            if completed_steps >= max_train_steps:
                break

    accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
    if accelerator.is_main_process:
        checkpoint_path = os.path.join(config.output_dir, f"steps_{completed_steps}")
        logger.info(f"Training finished. Final checkpoint saved at {checkpoint_path}")
      

def main():
    config = TrainingConfig()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train(config)

if __name__ == "__main__":
    main()
