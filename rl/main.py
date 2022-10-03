import gym
import numpy as np


def basic_policy(obs):
    array = obs[0]
    print(array.shape)
    print('---')
    angle = obs[0][2]
    return 0 if angle < 0 else 1


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    #print(env.action_space)

    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        for step in range(1000):
            action = basic_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards += reward
            print(terminated, truncated, episode_rewards)
            if terminated:
                break
        totals.append(episode_rewards)
    #print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
