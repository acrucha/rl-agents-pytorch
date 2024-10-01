import argparse
import os

import gymnasium as gym
import numpy as np
import rsoccer_gym
import torch

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy
from agents.utils.gif import generate_gif
from agents.ddpg import DDPGHP


def get_env_specs(env_name):
    env = gym.make(env_name)
    return env.observation_space.shape[0], env.action_space.shape[0], env.spec.max_episode_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="checkpoint to load")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    checkpoint = torch.load(args.checkpoint)

    env = gym.make(checkpoint['ENV_NAME'], render_mode="rgb_array")

    if checkpoint['AGENT'] == 'ddpg_async':
        pi = DDPGActor(checkpoint['N_OBS'], checkpoint['N_ACTS']).to(device)
    elif checkpoint['AGENT'] == 'sac_async':
        pi = GaussianPolicy(checkpoint['N_OBS'], checkpoint['N_ACTS'],
                            checkpoint['LOG_SIG_MIN'],
                            checkpoint['LOG_SIG_MAX'], checkpoint['EPSILON']).to(device)
    else:
        raise AssertionError
    
    hp = DDPGHP(
        EXP_NAME=checkpoint['ENV_NAME'],
        DEVICE=torch.device('cpu'),
        ENV_NAME=checkpoint['ENV_NAME'],
        N_ROLLOUT_PROCESSES=2,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=3,
        NOISE_SIGMA_INITIAL=0.8,
        NOISE_THETA=0.15,
        NOISE_SIGMA_DECAY=0.99,
        NOISE_SIGMA_MIN=0.15,
        NOISE_SIGMA_GRAD_STEPS=3000,
        REPLAY_SIZE=5000000,
        REPLAY_INITIAL=100000,
        SAVE_FREQUENCY=100000,
        GIF_FREQUENCY=100000,
        TOTAL_GRAD_STEPS=2000000
    )


    pi.load_state_dict(checkpoint['pi_state_dict'])
    pi.eval()

    generate_gif(env=env, filepath=args.checkpoint.replace(
        "pth", "gif").replace("checkpoint", "gif"), pi=pi, hp=hp) # , device=device)
