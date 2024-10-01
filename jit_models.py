from agents.ddpg.networks import DDPGActor
import gymnasium as gym
import torch
import sys

def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")
    return model

def jit_model(env_id, path):
    state_dict = load_model(f"{path}/actor.pth")
    env = gym.make(env_id)
    actor = DDPGActor(state_dict['N_OBS'], state_dict['N_ACTS']).to("cpu")
    actor.load_state_dict(state_dict['pi_state_dict'])
    actor.eval()
    obs, _ = env.reset()
    traced_script_module = torch.jit.trace(actor, torch.Tensor(obs.reshape(1, -1)))
    traced_script_module.save(f"{path}/actor_jit.pt")


if __name__ == "__main__":
    env_id = sys.argv[1] if len(sys.argv) > 1 else "Please provide an environment id"
    
    if len(sys.argv) < 2:
        print(env_id)
        sys.exit(0)

    if sys.argv[1] == "help":
        print("Usage: python jit_models.py <env_id> <path>")
        sys.exit(0)

    path = sys.argv[2] if len(sys.argv) > 2 else "Please provide a path to the model"

    if len(sys.argv) < 3:
        print(path)
        sys.exit(0)

    jit_model(env_id=env_id, path=path)
