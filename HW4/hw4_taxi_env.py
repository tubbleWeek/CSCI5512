import gymnasium as gym
env = gym.make("Taxi-v3", render_mode="rgb_array")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # sample a random action
    observation, reward, terminated, truncated, info = env.step(action)
    # print(action)
    if terminated or truncated:
        observation, info = env.reset()
    
# env.render()
env.close()