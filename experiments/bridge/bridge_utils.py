'''
Copied over from the OpenVLA eval code linked below, with some modifications:
https://github.com/openvla/openvla/blob/main/experiments/robot/bridge/widowx_env.py
'''

import os
import sys
import time
from typing import Dict

import gym
import imageio
import numpy as np
import PIL
import tensorflow as tf
import torch
from pyquaternion import Quaternion
from qwen_vl_utils import process_vision_info

from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

from collections import deque
import numpy as np

sys.path.append(".")

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})





    
def state_to_eep(xyz_coor, zangle: float):
    """
    Implements the state to end-effector pose function, returning a 4x4 matrix.
    Refer to `widowx_controller/widowx_controller.py` in the `bridge_data_robot` codebase.
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(
        matrix=DEFAULT_ROTATION
    )
    new_pose[:3, :3] = new_quat.rotation_matrix
    return new_pose


def wait_for_obs(widowx_client):
    """Fetches an observation from the WidowXClient."""
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs


def convert_obs(obs, im_size):
    """Preprocesses image and proprio observations."""
    # Preprocess image
    image_obs = (
        obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255
    ).astype(np.uint8)
    # Add padding to proprio to match RLDS training
    proprio = np.concatenate([obs["state"][:6], [0], obs["state"][-1:]])
    return {
        "image_primary": image_obs,
        "full_image": obs["full_image"],
        "proprio": proprio,
    }


def null_obs(img_size):
    """Returns a dummy observation with all-zero image and proprio."""
    return {
        "image_primary": np.zeros((img_size, img_size, 3), dtype=np.uint8),
        "proprio": np.zeros((8,), dtype=np.float64),
    }


class WidowXGym(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    """

    def __init__(
        self,
        widowx_client: WidowXClient,
        cfg: Dict,
        im_size: int = 256,
        blocking: bool = True,
    ):
        self.widowx_client = widowx_client
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "full_image": gym.spaces.Box(
                    low=np.zeros((480, 640, 3)),
                    high=255 * np.ones((480, 640, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((8,)) * -1, high=np.ones((8,)), dtype=np.float64
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64
        )
        self.cfg = cfg

    def step(self, action):
        self.widowx_client.step_action(action, blocking=self.blocking)

        raw_obs = self.widowx_client.get_observation()

        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True
            obs = null_obs(self.im_size)  # obs with all zeros
        else:
            obs = convert_obs(raw_obs, self.im_size)

        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.widowx_client.reset()
        self.move_to_start_state()

        raw_obs = wait_for_obs(self.widowx_client)
        obs = convert_obs(raw_obs, self.im_size)

        return obs, {}

    def get_observation(self):
        raw_obs = wait_for_obs(self.widowx_client)
        obs = convert_obs(raw_obs, self.im_size)
        return obs

    def move_to_start_state(self):
        successful = False
        while not successful:
            try:
                # Get XYZ position from user.
                init_x, init_y, init_z = self.cfg.init_ee_pos
                x_val = input(
                    f"Enter x value of gripper starting position (leave empty for default == {init_x}): "
                )
                if x_val == "":
                    x_val = init_x
                y_val = input(
                    f"Enter y value of gripper starting position (leave empty for default == {init_y}): "
                )
                if y_val == "":
                    y_val = init_y
                z_val = input(
                    f"Enter z value of gripper starting position (leave empty for default == {init_z}): "
                )
                if z_val == "":
                    z_val = init_z
                # Fix initial orientation and add user's commanded XYZ into start transform.
                # Initial orientation: gripper points ~15 degrees away from the standard orientation (quat=[0, 0, 0, 1]).
                transform = np.array(
                    [
                        [0.267, 0.000, 0.963, float(x_val)],
                        [0.000, 1.000, 0.000, float(y_val)],
                        [-0.963, 0.000, 0.267, float(z_val)],
                        [0.00, 0.00, 0.00, 1.00],
                    ]
                )
                # IMPORTANT: It is very important to move to reset position with blocking==True.
                #            Otherwise, the controller's `_reset_previous_qpos()` call will be called immediately after
                #            the move command is given -- and before the move is complete -- and the initial state will
                #            be totally incorrect.
                self.widowx_client.move(transform, duration=0.8, blocking=True)
                successful = True
            except Exception as e:
                print(e)


def get_widowx_env_params(cfg):
    """Gets (mostly default) environment parameters for the WidowX environment."""
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params["override_workspace_boundaries"] = cfg.bounds
    env_params["camera_topics"] = cfg.camera_topics
    env_params["return_full_image"] = True
    return env_params


def get_widowx_env(cfg, model=None):
    """Get WidowX control environment."""
    # Set up the WidowX environment parameters
    env_params = get_widowx_env_params(cfg)
    start_state = np.concatenate([cfg.init_ee_pos, cfg.init_ee_quat])
    env_params["start_state"] = list(start_state)
    # Set up the WidowX client
    widowx_client = WidowXClient(host=cfg.host_ip, port=cfg.port)
    widowx_client.init(env_params)
    env = WidowXGym(
        widowx_client,
        cfg=cfg,
        blocking=cfg.blocking,
    )
    return env


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input(
            "Enter the task name (or leave blank to repeat the previous task): "
        )
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_video(rollout_images, idx, task_label):
    """Saves an MP4 replay of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
    mp4_path = f"./rollouts/{task_label}-{idx+1}-{DATE_TIME}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def save_rollout_data(
    rollout_orig_images, rollout_images, rollout_states, rollout_actions, idx
):
    """
    Saves rollout data from an episode.

    Args:
        rollout_orig_images (list): Original rollout images (before preprocessing).
        rollout_images (list): Preprocessed images.
        rollout_states (list): Proprioceptive states.
        rollout_actions (list): Predicted actions.
        idx (int): Episode index.
    """
    os.makedirs("./rollouts", exist_ok=True)
    path = f"./rollouts/rollouts-{idx+1}-{DATE_TIME}.npz"
    # Convert lists to numpy arrays
    orig_images_array = np.array(rollout_orig_images)
    images_array = np.array(rollout_images)
    states_array = np.array(rollout_states)
    actions_array = np.array(rollout_actions)
    # Save to a single .npz file
    np.savez(
        path,
        orig_images=orig_images_array,
        images=images_array,
        states=states_array,
        actions=actions_array,
    )
    print(f"Saved rollout data at path {path}")


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(
        img, expand_animations=False, dtype=tf.uint8
    )  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_preprocessed_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    obs["full_image"] = resize_image(obs["full_image"], resize_size)
    return obs["full_image"]


def refresh_obs(obs, env):
    """Fetches new observations from the environment and updates the current observations."""
    new_obs = env.get_observation()
    obs["full_image"] = new_obs["full_image"]
    obs["image_primary"] = new_obs["image_primary"]
    obs["proprio"] = new_obs["proprio"]
    return obs

