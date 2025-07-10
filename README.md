# NORA: Neural Orchestrator for Robotics Autonomy

🔥 All the eval scripts and model checkpoints have been released.

🔥 Training scripts have been released.

🔥 Now supports Lerobot dataset format for training!

<div align="center">
  <img src="assets/nora-logo.png" alt="TangoFluxOpener" width="500" />
  
  [![Static Badge](https://img.shields.io/badge/nora-demos?label=nora-demos&link=http%3A%2F%2Fdeclare-lab.github.io%2Fnora)](http://declare-lab.github.io/nora) [![Static Badge](https://img.shields.io/badge/nora-checkpoints?label=nora-checkpoints&link=https%3A%2F%2Fhuggingface.co%2Fcollections%2Fdeclare-lab%2Fnora-6811ba3e820ef362d9eca281)](https://huggingface.co/collections/declare-lab/nora-6811ba3e820ef362d9eca281)  [![Static Badge](https://img.shields.io/badge/Read_the_paper-Arxiv?link=https%3A%2F%2Fwww.arxiv.org%2Fabs%2F2504.19854)](https://www.arxiv.org/abs/2504.19854)
  
</div>

## NORA in Action


https://github.com/user-attachments/assets/fe0384d9-b2eb-4ab0-b65a-a285ceb4b349


We are releasing some of the videos recorded during experiments showing how NORA performs real-world tasks with the WidowX robot -- [WidowX Demos](https://declare-lab.github.io/nora#demos).

## Checkpoints
[Model weights on Huggingface](https://huggingface.co/collections/declare-lab/nora-6811ba3e820ef362d9eca281)
## Getting Started For Inference
We provide a lightweight interface with minimal dependencies to get started with loading and running Nora for inference.
```bash
git clone https://github.com/declare-lab/nora.git
cd inference
# Create and activate conda environment
conda create -n nora python=3.10 -y
conda activate nora
pip install -r requirements.txt
```
For example, to load Nora for zero-shot instruction following in the BridgeData V2 environments with a WidowX robot:
```python

# Load VLA
from inference.nora import Nora
nora = Nora(device='cuda')

# Get Inputs
image: Image.Image = camera(...)
instruction: str = <INSTRUCTION>
# Predict Action (7-DoF; un-normalize for BridgeData V2)
actions = nora.inference(
    image=image,  # Dummy image
    instruction=instruction,
    unnorm_key='bridge_orig'  # Optional, specify if needed
)
# Execute...
robot.act(action, ...)
```

## How to Pretrain Nora/ Finetune nora in RLDSDataset
```bash
git clone https://github.com/declare-lab/nora.git
cd training
# Create and activate conda environment
conda create -n nora_train python=3.10 -y
conda activate nora_train
pip install -r requirements.txt
```
Our repository make use of huggingface's accelerate library for package from Hugging Face for multi-GPU training. Set up your own accelerator config base on your cluster's configuration. Model hyperparameters/settings are stored in the TrainingConfig in train.py. 
To download the dataset for training, you can refer to [Open X-Embodiment (OXE) mixture](https://robotics-transformer-x.github.io/) for details. Our dataset structure uses the same RLDS format used by [OpenVLA](https://github.com/openvla/openvla) training. You can also check OpenVLA's github for more information .
Once you have set the correct data path etcs, you can simply train nora with the following command!
```bash
accelerate launch --config_file='your_accelerator_accelerate_config.yaml train.py'
```
## ⚠️ Finetune with Action Chunking (Important)
To finetune NORA-LONG/NORA with different action horizon length, you will have to modify the future action window size as shown below https://github.com/declare-lab/nora/blob/5ad1658aa41c87e4cbb2f9da3f73b62840070280/training/datasets/datasets.py#L132. 

## Evaluating Nora on WidowX BridgeV2
We use OpenVLA's codebase to peform evaluation on Widow X BridgeV2. Please check OpenVLA's github repository on instructions how to set up WidowX robot server for BridgeData V2  evaluations. 
[https://github.com/openvla/openvla/tree/main?tab=readme-ov-file#evaluating-openvla](https://github.com/openvla/openvla/tree/main?tab=readme-ov-file#bridgedata-v2-widowx-evaluations)

After setting up the Widow X's robot server, you can open another terminal window to run the Nora policy evaluation script:
```python
cd experiments/bridge/
python run_widowx.py
```


## How to Pretrain Nora/ Finetune nora in Lerobot dataset
As of now, we are using a different version of torch when finetuning with Lerobot dataset due to Lerobot expects torchvision>=0.21.0.
```bash
git clone https://github.com/declare-lab/nora.git
cd lerobot_training
# Create and activate conda environment
conda create -n nora_lerobot python=3.10 -y
conda activate nora_lerobot
pip install -r lerobot_requirements.txt
```
Model hyperparameters/settings are stored in the TrainingConfig in lerobot_training.py. You can specify the path to the corresponding Lerobot dataset you wish to finetune Nora on. Do note that Nora is pretrained on 7 DoF action space (6+1 grippler action), finetuning on other action space may not work well. 

Note that in Nora's pretraining, grippler action is being flipped and normalized to [0, 1] (0 = close, 1 = open), whereas some dataset  such as LIBERO have (-1 = open, +1 = close). The invert_grippler_action flag in TrainingConfig will map grippler action from [-1,1] to [0,1].

You can also pass in a Lerobot Unnormalizer for action decoding
```
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.configs.types import  NormalizationMode, PolicyFeature
from lerobot.policies.normalize import (
    Unnormalize,
)

metadata = LeRobotDatasetMetadata('lerobot/libero_object_image')
stats = metadata.stats

features = {
            'action': PolicyFeature(shape=stats['action']['mean'].shape, type='action')
        }
norm_map = {
    'action': NormalizationMode.MIN_MAX,
}
unnormalize = Unnormalize(features=features, norm_map=norm_map, stats=stats)

image: Image.Image = camera(...)
instruction: str = <INSTRUCTION>
# Predict Action (7-DoF; un-normalize for BridgeData V2)
actions = nora.inference(
    image=image,  # Dummy image
    instruction=instruction,
    unnorm_key=None,  # Optional, specify if needed
    unnormalizer=unnormalize
)

# OPTIONAL. If your environment expect [-1,1] range and you train with invert_grippler_action=True, you will need to map back from [0,1] to [-1,1]
# actions = normalize_gripper_action(actions, binarize=True)
# actions = invert_gripper_action(actions)



```
## Acknowledgement
This repository is built based on [OpenVLA](https://github.com/openvla/openvla), [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment?tab=readme-ov-file),[transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate), [Qwen2.5 VL](https://github.com/QwenLM/Qwen2.5-VL). Thanks!

## Citation
```
@misc{hung2025norasmallopensourcedgeneralist,
      title={NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks}, 
      author={Chia-Yu Hung and Qi Sun and Pengfei Hong and Amir Zadeh and Chuan Li and U-Xuan Tan and Navonil Majumder and Soujanya Poria},
      year={2025},
      eprint={2504.19854},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.19854}, 
}
```


