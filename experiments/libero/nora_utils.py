import torch
import numpy as np
import PIL.Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from typing import Optional, Union, Dict, Any
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
import json

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

class Nora:
    
    # Define action token range and normalization bounds as class attributes
    # These are specific to the model's vocabulary and task
    _ACTION_TOKEN_MIN = 151665
    _ACTION_TOKEN_MAX = 153712


    def __init__(
        self,
        model_path: str ="declare-lab/nora",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16 # Make dtype configurable
    ):
        """
        Initializes the QwenRoboInference model and processors.

        Args:
            model_path (str): Hugging Face model ID or local path for the main
                            Qwen 2.5 VL model and processor.
            device (Optional[str]): The device to use for inference (e.g., "cuda:0", "cpu").
                                     If None, automatically detects CUDA or falls back to CPU.
            torch_dtype (torch.dtype): The data type to use for the model.
                                       Defaults to torch.bfloat16.
        Raises:
            RuntimeError: If models or processors fail to load, or device is unavailable.
        """
        # --- Device Setup ---
        self.libero_keys= {'libero_object':[np.array( [
        -0.5383928418159485,
        -0.8758928775787354,
        -0.9375,
        -0.06964285671710968,
        -0.11678571254014969,
        -0.15964286029338837,
        0.0
      ]),np.array([
        0.8464285731315613,
        0.84375,
        0.9375,
        0.08142857253551483,
        0.14892856776714325,
        0.0867857113480568,
        1.0
      ])],'libero_object_min':[np.array([-0.8839286 , -0.9375    , -0.9375    , -0.15000001, -0.29035714,
        -0.32892856, -1.        ]),np.array([0.9375    , 0.89196426, 0.9375    , 0.17678571, 0.35035715,
        0.18107143, 1.        ])],
      'libero_10':[np.array([
        -0.6348214149475098,
        -0.7741071581840515,
        -0.7633928656578064,
        -0.09749999642372131,
        -0.14819999992847435,
        -0.2742857038974762,
        0.0
      ]),np.array([
        0.7714285850524902,
        0.8464285731315613,
        0.9375,
        0.13928571343421936,
        0.15964286029338837,
        0.3246428668498993,
        1.0
      ])],'libero_90':[np.array([
        -0.6348214149475098,
        -0.7741071581840515,
        -0.7633928656578064,
        -0.09749999642372131,
        -0.14819999992847435,
        -0.2742857038974762,
        0.0
      ]),np.array([
        0.7714285850524902,
        0.8464285731315613,
        0.9375,
        0.13928571343421936,
        0.15964286029338837,
        0.3246428668498993,
        1.0
      ])],'libero_goal':[np.array([-0.8785714507102966,
        -0.7553571462631226,
        -0.9375,
        -0.1510714292526245,
        -0.1639285683631897,
        -0.13777500048279764,
        0.0])
       ,np.array([0.9375,
        0.9107142686843872,
        0.9375,
        0.20357142388820648,
        0.26357144117355347,
        0.375,
        1.0])],'libero_spatial':[np.array([
        -0.7454732114076613,
        -0.6616071462631226,
        -0.9375,
        -0.1071428582072258,
        -0.20678570866584778,
        -0.1842857152223587,
        0.0
      ]),np.array([
        0.9375,
        0.8758928775787354,
        0.9321428537368774,
        0.1039285734295845,
        0.17678570747375488,
        0.14571428298950195,
        1.0
      ])],
      'libero_spatial_min':[np.array([-0.9375    , -0.9375    , -0.9375    , -0.1875    , -0.36750001,
        -0.36000001, -1]),
        np.array([0.9375    , 0.9375    , 0.9375    , 0.19714285, 0.33642858,
        0.375     , 1.        ])]}

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")

        if self.device.startswith("cuda"):
             if not torch.cuda.is_available():
                  raise RuntimeError(f"CUDA is not available, but device '{self.device}' was specified.")
             gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
             if gpu_id >= torch.cuda.device_count():
                  raise RuntimeError(f"CUDA device {gpu_id} not available. Only {torch.cuda.device_count()} devices found.")

        # --- Load Fast Tokenizer ---
        try:
            print(f"Loading fast tokenizer from: physical-intelligence/fast")
            self.fast_tokenizer = AutoProcessor.from_pretrained(
                "physical-intelligence/fast", trust_remote_code=True
            )
            # Ensure required attributes are set/exist
           
            self.fast_tokenizer.action_dim = 7 # Set default if not in config
            print("Setting action_dim  to 7.")
           
            self.fast_tokenizer.time_horizon = 1 # Set default if not in config
            print("Setting time horizon to 1.")

        except Exception as e:
            raise RuntimeError(
                f"Error loading fast tokenizer: {e}. "
            )

        # --- Load Main Processor ---
        try:
            print(f"Loading main processor from: {model_path}")
            # Assuming the main processor is saved in the same location as the model
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(f"Error loading main processor from {model_path}: {e}")

        # --- Load Main Model ---
        try:
            print(f"Loading model from: {model_path}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            #    attn_implementation="flash_attention_2", # Comment out this line if there is an error with flash attention
            )
            self.model.to(self.device)
            # Load generation config and set specific parameters
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
            self.model.generation_config.do_sample = False # Ensure deterministic output

            self.model.eval() # Set the model to evaluation mode

            repo_id = "declare-lab/nora"
            filename = "norm_stats.json"

            # Download the norm_stats locally (only downloads once; cached)
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)

            # Load the JSON file
            with open(file_path, "r") as f:
                norm_stats = json.load(f)
            self.norm_stats = norm_stats
            

        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")

        print("Model and processors loaded successfully.")
    
    @torch.inference_mode()
    def inference(self, image: np.ndarray, instruction: str,unnorm_key: str = None,unnormalizer=None) -> np.ndarray:
        """
        Performs inference to get robotic actions based on an image and instruction.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C).
            instruction (str): The natural language instruction.
            unnorm_key (str, optional): Key to select normalization statistics for unnormalizing actions.
            unnormalizer (lerobot.policies.normalize.Unnormalize, optional): If a Lerobot Unnormalizer is provided, it will be used to unnormalize the action.

        Returns:
            np.ndarray: The predicted unnormalized robotic action array.
        """
        # --- Prepare Inputs ---
        # Ensure image is PIL Image for processor compatibility
        if not isinstance(image, PIL.Image.Image):
             image = PIL.Image.fromarray(image)

        # Construct messages in the expected chat format. Note that nora expects image of size 224 by 224
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # Apply chat template to get the text input for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision information (depends on your process_vision_info function)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model using the main processor
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # --- Generate Output ---
        
        generated_ids = self.model.generate(**inputs)

    

        # --- Extract and Decode Action ---
        # Find the indices of tokens within the action token range
       
        
        start_idx = (self._ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= self._ACTION_TOKEN_MAX)
        start_idx = torch.where(start_idx)[0]

        if len(start_idx) > 0:
            start_index = start_idx[0].item()
        else:
            start_index = None  # or -1 to indicate not found


        # Extract the first action token ID

        # Decode the action token using the fast tokenizer
        # The token ID needs to be map back to the range expected by the fast tokenizer decoder

        
       
        output_action = self.fast_tokenizer.decode([generated_ids[0][start_idx] - self._ACTION_TOKEN_MIN])
        
        if unnormalizer is not None: ## If a Lerobot Unnormalizer is provided, use it to unnormalize the action
            #
            unnormalized_action = unnormalizer({'action':output_action})
            return unnormalized_action['action']
       

        # --- Denormalize Action ---
        # Assuming output_action is a numpy array of shape (1, time_horizon, action_dim)
        # and the values are in the range [-1, 1]
        # The formula is: unnormalized = 0.5 * (normalized + 1) * (high - low) + low

        '''We use the norm stats computed from OpenVLA https://arxiv.org/abs/2406.09246'''
        if 'libero' in unnorm_key:
            action_low,action_high = self.libero_keys[unnorm_key]
        else:
            action_norm_stats = self.get_action_stats(unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

        unnorm_actions = (
            0.5 * (output_action + 1) * (action_high - action_low)
            + action_low
        )

      

        
        return np.array(unnorm_actions[0])

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
    