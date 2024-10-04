import argparse
import os
import time

import gymnasium as gym
import numpy as np
import rsoccer_gym
import torch

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="checkpoint to load")
    
    parser.add_argument("-s", "--steps-per-ep", default=1200, type=int)
    parser.add_argument("-e", "--episodes", default=100, type=int)
    parser.add_argument("-t", "--number-of-tests", default=30, type=int)
    parser.add_argument("--goto", default=False)

    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    checkpoint = torch.load(args.checkpoint)

    env = gym.make(checkpoint['ENV_NAME'], render_mode='human')

    if checkpoint['AGENT'] == 'ddpg_async':
        pi = DDPGActor(checkpoint['N_OBS'], checkpoint['N_ACTS']).to(device)
    elif checkpoint['AGENT'] == 'sac_async':
        pi = GaussianPolicy(checkpoint['N_OBS'], checkpoint['N_ACTS'],
                            checkpoint['LOG_SIG_MIN'],
                            checkpoint['LOG_SIG_MAX'], checkpoint['EPSILON']).to(device)
    else:
        raise AssertionError

    pi.load_state_dict(checkpoint['pi_state_dict'])
    pi.eval()  

    total_tests = args.number_of_tests
    total_goals_percentage = 0

    while total_tests > 0:
        goals = 0
        total_tests -= 1
        total_episodes = args.episodes
        while total_episodes > 0:
            total_episodes -= 1
            done = False
            s, _ = env.reset()
            info = {}
            ep_steps = 0
            ep_rw = 0
            st_time = time.perf_counter()
            for i in range(args.steps_per_ep):
                # Step the environment
                s_v = torch.Tensor(s).to(device)
                a = pi.get_action(s_v)
                s_next, r, done, trunc, info = env.step(a)
                ep_steps += 1
                ep_rw += r
                if done:
                    break

                # Set state for next step
                s = s_next

            info['fps'] = ep_steps / (time.perf_counter() - st_time)
            info['ep_steps'] = ep_steps
            info['ep_rw'] = ep_rw

            if not args.goto:
                if total_episodes % 10 == 0:
                    print("Episode done in %d steps, reward %.3f, FPS %.2f, goal_score %d" % (
                        ep_steps, ep_rw, info['fps'], info['goal_score']))
                goals += info['goal_score']
        if not args.goto:
            total_goals_percentage += (goals / args.episodes)
            print("Test %d - Episode %d - Goal percentage: %.2f" % (args.number_of_tests - total_tests, args.episodes - total_episodes, goals / args.episodes))

    print("-----------FINISHED-----------")
    if not args.goto:
        print("Mean goal percentage: %.2f" % (total_goals_percentage / args.number_of_tests))