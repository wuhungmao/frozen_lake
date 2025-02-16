from frozen_lake import Frozen_lake_wrapper
import gymnasium as gym
def test_return_reward():
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=10, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.return_reward(1) == 0)
    assert(fro_zen_mdp.return_reward(19) == -1)
    assert(fro_zen_mdp.return_reward(29) == -1)
    assert(fro_zen_mdp.return_reward(0) == 0)
    assert(fro_zen_mdp.return_reward(63) == 1)
    

def test_find_successors():
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=10, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.find_successors(0,0) == [None, None, 8])
    assert(fro_zen_mdp.find_successors(0,1) == [None, 8, 1])
    assert(fro_zen_mdp.find_successors(0,2) == [8, 1, None])
    assert(fro_zen_mdp.find_successors(0,0) == [None, None, 8])
    assert(fro_zen_mdp.find_successors(1,0) == [None, 0, 9])
    assert(fro_zen_mdp.find_successors(1,1) == [0, 9, 2])

def test_calc_backup_val():
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=10, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.calc_backup_val(62)[1] == 1)
    assert(fro_zen_mdp.calc_backup_val(0)[1] == 0)
    assert(fro_zen_mdp.calc_backup_val(11)[1] == 3)
    assert(fro_zen_mdp.calc_backup_val(18)[1] == 0)
    return None

