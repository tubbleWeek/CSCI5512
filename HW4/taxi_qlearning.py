from collections import defaultdict
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import sys
import random
'''
Based on agent from https://gymnasium.farama.org/introduction/train_agent/
'''
class TaxiQLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        
        self.env = env
        self.q_values = defaultdict(self.zero_array)

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    def zero_array(self):
            return np.zeros(self.env.action_space.n)
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def get_qvals(self):
        return self.q_values

def main(exploration_param, learning_rt, discount_f):
    # hyperparameters
    learning_rate = learning_rt
    n_episodes = 100_00
    start_epsilon = exploration_param
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1
    env = gym.make("Taxi-v3")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = TaxiQLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_f
    )

    '''
    Train agent
    '''
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
        # agent.decay_epsilon()
    '''
    Write learned policy to pickle
    '''

    # dict with dict as value
    with open("qlearning_q_vals.pickle", "wb") as f:
        q_vals = agent.get_qvals()
        q_vals = pickle.dumps(q_vals)
        pickle.dump(q_vals, f)


    # just dict
    with open("qlearning_policy.pickle", "wb") as f:
        q_vals = agent.get_qvals()
        for state in q_vals:
            pickle.dump(q_vals[state], f)
    
    '''Random Agent'''
    rand_env = gym.make("Taxi-v3")
    rand_env = gym.wrappers.RecordEpisodeStatistics(rand_env, buffer_length=n_episodes)
    for episode in tqdm(range(n_episodes)):
        obs, info = rand_env.reset()
        done = False
        # print(rand_env.action_space.sample())
        # play one episode
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = rand_env.step(action)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

    '''
    Ploting pretty graphs
    '''
    def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a 500 episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    axs[0].set_title("Q-Learning Agent rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    # axs[1].set_title("Episode lengths")
    # length_moving_average = get_moving_avgs(
    #     env.length_queue,
    #     rolling_length,
    #     "valid"
    # )
    # axs[1].plot(range(len(length_moving_average)), length_moving_average)

    # axs[2].set_title("Training Error")
    # training_error_moving_average = get_moving_avgs(
    #     agent.training_error,
    #     rolling_length,
    #     "same"
    # )
    # axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[1].set_title("Random Agent Rewards")
    random_enviroment_rewards = get_moving_avgs(
        rand_env.return_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(random_enviroment_rewards)), random_enviroment_rewards)
    plt.tight_layout()
    # plt.show()
    plt.savefig("qlearning total reward.png", transparent=False)

if __name__ == "__main__":
    exploration_parameter = (float)(sys.argv[1])
    learning_rate = (float)(sys.argv[2])
    discount_factor = (float)(sys.argv[3])
    if exploration_parameter < 0 or exploration_parameter > 1:
        print("exploration parameter not between (0,1)")
        exit()
    if learning_rate < 0 or learning_rate > 1:
        print("learning parameter not between (0,1)")
        exit()
    if discount_factor < 0 or discount_factor > 1:
        print("discount parameter not between (0,1)")
        exit()
    # print(exploration_parameter +" " + learning_rate + " " + discount_factor)
    main(exploration_parameter, learning_rate, discount_factor)