# NORA: Neural Orchestration for Robotics Autonomy

🔥 Training code and more examples will be released soon.

<div align="center">
  <img src="assets/nora-logo.png" alt="TangoFluxOpener" width="500" />
  
  [![Static Badge](https://img.shields.io/badge/nora-demos?label=nora-demos&link=http%3A%2F%2Fdeclare-lab.github.io%2Fnora)](http://declare-lab.github.io/nora) [![Static Badge](https://img.shields.io/badge/nora-checkpoints?label=nora-checkpoints&link=https%3A%2F%2Fhuggingface.co%2Fcollections%2Fdeclare-lab%2Fnora-6811ba3e820ef362d9eca281)](https://huggingface.co/collections/declare-lab/nora-6811ba3e820ef362d9eca281)  [![Static Badge](https://img.shields.io/badge/Read_the_paper-Arxiv?link=https%3A%2F%2Fwww.arxiv.org%2Fabs%2F2504.19854)](https://www.arxiv.org/abs/2504.19854)
  
</div>

# NORA in Action
We are releasing some of the videos recorded during experiments showing how NORA performs real-world tasks with the WidowX robot -- [WidowX Demos](https://declare-lab.github.io/nora#demos).

# Checkpoints
[Model weights on Huggingface](https://huggingface.co/collections/declare-lab/nora-6811ba3e820ef362d9eca281)
# Getting Started For Inference
We provide a lightweight interface with minimal dependencies to get started with loading and running Nora for inference.
```bash
cd inference
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


