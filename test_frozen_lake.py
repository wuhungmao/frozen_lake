from frozen_lake import Frozen_lake_wrapper
import gymnasium as gym
import pytest
def test_return_reward():
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.return_reward(1) == 0)
    assert(fro_zen_mdp.return_reward(19) == -1)
    assert(fro_zen_mdp.return_reward(29) == -1)
    assert(fro_zen_mdp.return_reward(0) == 0)
    assert(fro_zen_mdp.return_reward(63) == 1)
    
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=False)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.return_reward(1) == 0)
    assert(fro_zen_mdp.return_reward(19) == -1)
    assert(fro_zen_mdp.return_reward(29) == -1)
    assert(fro_zen_mdp.return_reward(0) == 0)
    assert(fro_zen_mdp.return_reward(63) == 1)
    

def test_find_successors():
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.find_successors(0,0) == [None, None, 8])
    assert(fro_zen_mdp.find_successors(0,1) == [None, 8, 1])
    assert(fro_zen_mdp.find_successors(0,2) == [8, 1, None])
    assert(fro_zen_mdp.find_successors(1,0) == [None, 0, 9])
    assert(fro_zen_mdp.find_successors(1,1) == [0, 9, 2])
    
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=False)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.find_successors(0,0) == [None])
    assert(fro_zen_mdp.find_successors(0,1) == [8])
    assert(fro_zen_mdp.find_successors(0,2) == [1])
    assert(fro_zen_mdp.find_successors(0,3) == [None])
    assert(fro_zen_mdp.find_successors(1,0) == [0])
    assert(fro_zen_mdp.find_successors(1,1) == [9])
    assert(fro_zen_mdp.find_successors(10,1) == [18])

def test_calc_backup_val():
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.calc_backup_val(62) == pytest.approx(0.33333333, rel=1e-2))
    assert(fro_zen_mdp.calc_backup_val(0) == 0)
    assert(fro_zen_mdp.calc_backup_val(11) == 0)
    assert(fro_zen_mdp.calc_backup_val(18) == 0)
    
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=False)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.calc_backup_val(62) == pytest.approx(.9, rel=1e-2))
    assert(fro_zen_mdp.calc_backup_val(0) == 0)
    assert(fro_zen_mdp.calc_backup_val(11) == -.1)
    assert(fro_zen_mdp.calc_backup_val(46) == -.1)
    assert(fro_zen_mdp.calc_backup_val(54) == -.1)
    

def test_sweep():
    env = gym.make('FrozenLake-v1', render_mode="human", max_episode_steps=100, disable_env_checker=False, map_name="8x8", is_slippery=True)
    fro_zen_mdp = Frozen_lake_wrapper(env)
    assert(fro_zen_mdp.sweep() == pytest.approx(0.33333333, rel=1e-2))
