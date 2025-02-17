# 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
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from frozen_lake import Frozen_lake_wrapper
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
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
        
    
