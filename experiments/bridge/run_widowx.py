import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
from experiments.nora_utils import Nora
import draccus
from bridge_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,

)

import numpy as np



### Our bridge inference code is adapted from OpenVLA codebase

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    

              # Pretrained checkpoint path
    
    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)
    long: bool=False
    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "localhost"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 200                                         # Max number of timesteps per episode
    control_frequency: float = 5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)

    # fmt: on


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
   
    assert (
        not cfg.center_crop
    ), "`center_crop` should be disabled for Bridge evaluations!"

    # Initialize the WidowX environment
    env = get_widowx_env(cfg)
    print("Env finished")

    # Get expected image dimensions
    resize_size = 224

    # Load model
   
    if cfg.long:

        nora = Nora('declare-lab/nora-long')
        nora.fast_tokenizer.time_horizon = 5 ## Set the fast tokenizer time horizon to 5
    else:
        nora = Nora('declare-lab/nora')


    

    

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        break_flag=1
        while t < cfg.max_steps:
            try:
            
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    
                        print(f"t: {t}")
                        print(
                            f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}"
                        )
                        last_tstamp = time.time()

                        # Refresh the camera image and proprioceptive state
                        obs = refresh_obs(obs, env)

                        # Save full (not preprocessed) image for replay video
                        replay_images.append(obs["full_image"])

                        # Get preprocessed image
                        obs["full_image"] = get_preprocessed_image(obs, resize_size)

                        # Query model to get action

                        #print("Proprio:",obs["proprio"])
                        
                        
                        
                        action = nora.inference(obs["full_image"], task_label,unnorm_key="bridge_orig")

                    
                        # [If saving rollout data] Save preprocessed image, robot state, and action
                        if cfg.save_data:
                            rollout_images.append(obs["full_image"])
                            rollout_states.append(obs["proprio"])
                            rollout_actions.append(action)

                        # Execute action
                        # for act in action:
                        #     obs, _, _, _, _ = env.step(act)
                        obs, _, _, _, _ = env.step(action[0])
                        
                        t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                    
                else:
                    print(f"\nCaught exception: {e}")
                save_rollout_video(replay_images, episode_idx, task_label)
                break_flag=0
                break

        # Save a replay video of the episode
        if break_flag==1:
            save_rollout_video(replay_images, episode_idx, task_label)

        # [If saving rollout data] Save rollout data

        if cfg.save_data:
            save_rollout_data(
                replay_images,
                rollout_images,
                rollout_states,
                rollout_actions,
                idx=episode_idx,
               
            )

        # Redo episode or continue
        if (
            input(
                "Enter 'r' if you want to redo the episode, or press Enter to continue: "
            )
            != "r"
        ):
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
