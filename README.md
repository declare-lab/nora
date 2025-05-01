# NORA: Neural Orchestration for Robotics Autonomy

🔥 Training scripts have been released.

<div align="center">
  <img src="assets/nora-logo.png" alt="TangoFluxOpener" width="500" />
  
  [![Static Badge](https://img.shields.io/badge/nora-demos?label=nora-demos&link=http%3A%2F%2Fdeclare-lab.github.io%2Fnora)](http://declare-lab.github.io/nora) [![Static Badge](https://img.shields.io/badge/nora-checkpoints?label=nora-checkpoints&link=https%3A%2F%2Fhuggingface.co%2Fcollections%2Fdeclare-lab%2Fnora-6811ba3e820ef362d9eca281)](https://huggingface.co/collections/declare-lab/nora-6811ba3e820ef362d9eca281)  [![Static Badge](https://img.shields.io/badge/Read_the_paper-Arxiv?link=https%3A%2F%2Fwww.arxiv.org%2Fabs%2F2504.19854)](https://www.arxiv.org/abs/2504.19854)
  
</div>

## NORA in Action
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

## How to Pretrain Nora/ Finetune nora
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
accelerate launch train.py --config_file='your_accelerator_accelerate_config.yaml'
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


