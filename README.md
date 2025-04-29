# NORA: Neural Orchestration for Robotics Autonomy.

Training code and more examples will be released soon.


# Getting Started For Inference
To get started with loading and running Nora for inference, we provide a lightweight interface that with minimal dependencies.
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




