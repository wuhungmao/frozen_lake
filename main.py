# 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
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from frozen_lake import Frozen_lake_wrapper
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="frozen lake")
    parser.add_argument("--map_size", type=str, default="8x8", choices=["8x8", "4x4"], help="4x4 or 8x8")
    parser.add_argument("--slippery", type=int, default=0, choices=[1, 0], help="0 is false, 1 is true")
    args = parser.parse_args()
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name=args.map_size, is_slippery=args.slippery)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    print(env.spec)
    fro_zen_mdp.env.reset()
    threshold = .000001
    delta = 1
    # running sweeps until value function converges
    while delta > threshold:
        delta = fro_zen_mdp.sweep()
        print("delta is: ", delta)
        fro_zen_mdp.dump()
        
    fro_zen_mdp.extra_opt_policy()
    print("optimal policy:", fro_zen_mdp.opt_policy)
    step = fro_zen_mdp.env.step(fro_zen_mdp.opt_policy[0].value)
    while step[2] != True and step[3] != True:
        step = fro_zen_mdp.env.step(fro_zen_mdp.opt_policy[step[0]].value)
        fro_zen_mdp.env.render()

    if step[2] == True:
        print("terminated")
    else:
        print("truncated")
        
    
