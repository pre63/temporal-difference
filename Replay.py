import numpy as np
import gymnasium as gym
from datetime import datetime

import sys
import json
import importlib


def save_config(path, model_path, model_class, env_module, env_name):
  """
  Save the configuration to a JSON file.

  Parameters:
      path (str): The directory where the configuration file will be saved.
      model_path (str): The Python module path to the model implementation.
      model_class (str): The name of the model class.
      env_module (str): The Python module path to the environment implementation.
      env_name (str): The name of the environment creation function.
  """
  print(f"Model Path: {model_path}, Model Class: {model_class}, Env Module: {env_module}, Env Name: {env_name}")

  config = {
      "model_path": model_path,
      "model_class": model_class,
      "env_module": env_module,
      "env_name": env_name
  }

  config_file = f"{path}/config.json"

  with open(config_file, "w") as f:
    json.dump(config, f, indent=4)

  print(f"Configuration saved to {config_file}")
  model_file_path = f"{path}/model"
  return model_file_path


def replay(env, top_model):

  done = False
  state, _ = env.reset()
  while not done:
    action = top_model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
  env.close()
  print("Done")


if __name__ == "__main__":
    # Argument: Path to the save folder
  path = sys.argv[1]

  model_file = path + "/model"
  config_file = path + "/config.json"
  config = json.load(open(config_file, "r"))

  # Load configuration details
  model_path = config["model_path"]  # Path to the module containing the model implementation
  model_class = config["model_class"]  # Name of the model class
  env_name = config["env_name"]  # Name of the environment creation function
  env_module = config["env_module"]  # Python file containing the environment implementation

  # Dynamically import the model and environment modules
  print(f"Loading model from {model_path}")
  model_module = importlib.import_module(model_path)
  ModelClass = getattr(model_module, model_class)

  env_module = importlib.import_module(env_module)
  env_function = getattr(env_module, env_name)
  
  # Initialize environment
  env = env_function(render_mode="human")

  # Initialize model and load weights
  model = ModelClass(observation_space=env.observation_space, action_space=env.action_space)
  model.load(model_file)


  # Replay the model in the environment
  replay(env, model)
