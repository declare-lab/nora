# NORA: Neural Orchestration for Robotics Autonomy.

Training code and more examples will be released soon.

# NORA in Action
We are releasing some of the videos recorded during experiments showing how NORA performs real-world tasks with the WidowX robot -- [WindowX Demos](https://declare-lab.github.io/nora).

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




