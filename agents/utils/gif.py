import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def generate_gif(
    env, 
    filepath, 
    pi, 
    hp, 
    max_episode_steps=1200, 
    resize_to=None, 
    duration=25
):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment
    filepath : str
    pi : nn.Module
    max_episode_steps : int
    resize_to : tuple of ints, optional
    duration : float, optional
    """
    
    # collect frames
    frames = []
    s, _ = env.reset()
    for t in tqdm(iterable=range(max_episode_steps), desc="Generating gif"):
        if hp.AGENT != "maddpg_async":
            s_v = torch.Tensor(s).to(hp.DEVICE)
            a = pi.get_action(s_v)
            s_next, r, done, trunc, info = env.step(a)
        else:
            a = [agent.action(obs) for agent, obs in zip(pi, s)]
            s_next, r, done, trunc, info = env.step(a)
        # store frame
        frame = env.render()
        frame = Image.fromarray(frame)
        frame = frame.convert('P', palette=Image.ADAPTIVE)
        if resize_to is not None:
            if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
                raise TypeError(
                    "expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = env.render()
    frame = Image.fromarray(frame)
    frame = frame.convert('P', palette=Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    frames[0].save(
        fp=filepath, 
        format='GIF', 
        append_images=frames[1:], 
        save_all=True,
        duration=duration, 
        loop=0
    )
