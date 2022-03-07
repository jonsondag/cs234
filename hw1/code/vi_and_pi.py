### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""

	value_function = np.zeros(nS)

	############################
	# YOUR IMPLEMENTATION HERE #
	P_tensor, R_tensor = get_tensors(P, nS, nA)
	P_matrix = P_tensor[np.arange(nS), policy, :]
	R_matrix = R_tensor[np.arange(nS), policy, :]
	rewards = np.sum(P_matrix * R_matrix, axis=1)
	stop = False
	i = 0
	while(stop is False):
		prev_value_function = value_function
		disc_value = np.dot(P_matrix, gamma * value_function)
		value_function = rewards + disc_value
		if (np.max(value_function - prev_value_function) < tol):
			stop = True
		print('policy evaluations: {}'.format(i))
		i += 1
	############################
	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	P_tensor, R_tensor = get_tensors(P, nS, nA)
	rewards = np.sum(P_tensor * R_tensor, axis=2)
	disc_value = np.dot(P_tensor, gamma * value_from_policy)
	state_action_function = rewards + disc_value
	new_policy = np.argmax(state_action_function, axis=1)
	############################
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	############################
	# YOUR IMPLEMENTATION HERE #
	stop = False
	i = 0
	while not stop:
		value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
		new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
		if np.max(new_policy - policy) < tol:
			stop = True
		policy = new_policy
		print('policy iterations: {}'.format(i))
		i += 1
	############################
	return value_function, policy


def get_tensors(P, nS, nA):
	"""
	Convert P dict to tensors

	Parameters:
	----------
	P, nS, nA:
		defined at beginning of file
	Returns:
	----------
	P_tensor: np.ndarray[nS, nA, nS].  Tensor indicating probability of transferring from start to finish states given action
	R_tensor: np.ndarray[nS, nA, nS].  Tensor indicating rewards corresponding to transfer from start to finish states given action
	"""
	P_tensor = np.zeros([nS, nA, nS])
	R_tensor = np.zeros([nS, nA, nS])
	for start_state_idx in range(len(P_tensor)):
		start_state = P[start_state_idx]
		for action_idx in range(len(start_state)):
			outcome_list = start_state[action_idx]
			for outcome in outcome_list:
				(probability, next_state_idx, reward, _) = outcome
				P_tensor[start_state_idx, action_idx, next_state_idx] = probability
				R_tensor[start_state_idx, action_idx, next_state_idx] = reward
	return P_tensor, R_tensor


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	P_tensor, R_tensor = get_tensors(P, nS, nA)
	rewards = np.sum(P_tensor * R_tensor, axis=2)
	stop = False
	i = 0
	while stop is False:
		disc_value = np.dot(P_tensor, gamma * value_function)
		state_action_function = rewards + disc_value
		policy = np.argmax(state_action_function, axis=1)
		value_function_old = value_function
		value_function = np.max(state_action_function, axis=1)
		if np.max(value_function - value_function_old) < tol:
			stop = True
		print('value iterations: {}'.format(i))
		i += 1
		print(value_function)
		print(policy)
	############################
	return value_function, policy


def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print(env.P)
	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	print(V_pi)
	print(p_pi)
	render_single(env, p_pi, 100)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	print(V_vi)
	print(p_vi)
	render_single(env, p_vi, 100)

	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	print(env.P)

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	print(V_pi)
	print(p_pi)
	render_single(env, p_pi, 100)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	print(V_vi)
	print(p_vi)
	render_single(env, p_vi, 100)
