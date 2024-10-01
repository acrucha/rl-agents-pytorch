import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="Pid-v0",
    entry_point="envs.pid:VSSPIDTuningEnv",
    kwargs={"max_steps": 1200, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="Penalty-v0",
    entry_point="envs.penalty:VSSPenaltyEnv",
    kwargs={"max_steps": 1200, "repeat_action": 1},
    max_episode_steps=1200,
)

register(
    id="Attacker-v0",
    entry_point="envs.vssef:VSSEF"
)

register(
    id="VSS-v0",
    entry_point="rsoccer_gym.vss.env_vss:VSSEnv",
)

# Using VSS Single Agent env
env = gym.make('VSS-v0', render_mode="human")

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
    print(reward)