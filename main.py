# 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
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from frozen_lake import Frozen_lake_wrapper
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=10, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    print(fro_zen_mdp.env.reset())
    print(fro_zen_mdp.env.step(2))
    print(fro_zen_mdp.env.render())
    print(fro_zen_mdp.env.step(1))
    print(fro_zen_mdp.env.render())
    print(fro_zen_mdp.action_spac)
    print(fro_zen_mdp.state_spac)
    print(fro_zen_mdp.val_func)
    print(fro_zen_mdp.env.spec)
    
    threshold = .1
    delta = 1
    # a sweep
    for state in fro_zen_mdp.state_spac:
        fro_zen_mdp.val_func[state] = fro_zen_mdp.calc_backup_val(state)
    
    print(fro_zen_mdp.val_func)
    # gym.register(id="FrozenLake-v1", entry_point=None, reward_threshold=10.0, nondeterministic=False, 
    # max_episode_steps=10, order_enforce=True, PassiveEnvChecker=True, vector_entry_point=None)
    
    # for curr_s in STATE_SPAC:
    #     val_func[curr_s] = cal_backup_val(curr_s)
    
