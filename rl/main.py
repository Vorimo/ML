import numpy as np
import progressbar
import gym
from tensorflow.python.keras.optimizer_v2.adam import Adam

from rl.agent import Agent

# note: for 08.10.2022 requires a stable_baselines3 package to be installed
if __name__ == '__main__':
    # prepare an environment
    environment = gym.make("Taxi-v3")
    environment.reset()
    environment.render()

    print('Number of states: {}'.format(environment.observation_space.n))
    action_space = environment.action_space
    print('Number of actions: {}'.format(action_space.n))

    optimizer = Adam(learning_rate=0.01)
    agent = Agent(environment, optimizer)

    batch_size = 2000  # 2000
    num_of_episodes = 20  # 20
    time_steps_per_episode = 100  # 100
    agent.q_network.summary()

    # training (could take 30m+, increase batch_size/num_of_episodes, decrease time_steps_per_episode to speed up)
    for e in range(0, num_of_episodes):
        # Reset the environment
        state = environment.reset()
        state = np.reshape(state, [1, 1])

        # Initialize variables
        reward = 0
        terminated = False

        bar = progressbar.ProgressBar(maxval=time_steps_per_episode / 10,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for timestep in range(time_steps_per_episode):
            # Run Action
            action = agent.act(environment, state)

            # Take action
            next_state, reward, terminated, info = environment.step(action)
            next_state = np.reshape(next_state, [1, 1])
            agent.store(state, action, reward, next_state, terminated)

            state = next_state

            # average termination is on the 20th step
            if terminated:
                agent.align_target_model()
                break

            if len(agent.experience_replay) > batch_size:
                agent.retrain(batch_size)

            if timestep % 10 == 0:
                bar.update(timestep / 10 + 1)

        bar.finish()
        if (e + 1) % 10 == 0:
            print("**********************************")
            print("Episode: {}".format(e + 1))
            environment.render()
            print("**********************************")
    print("Model trained!")

    # model evaluation (could take 30m+, decrease test_num_of_episodes to speed up)
    test_total_epochs = 0
    test_total_penalties = 0
    test_num_of_episodes = 15  # 100

    for _ in range(test_num_of_episodes):
        print(f'Episode {_}')
        state = environment.reset()
        state = np.reshape(state, [1, 1])
        epochs = 0
        penalties = 0
        reward = 0

        terminated = False

        while not terminated:
            if epochs % 10 == 0:
                print(f'Timestep {epochs}')
            action = np.argmax(agent.q_network.predict(state)[0])
            state, reward, terminated, info = environment.step(action)
            state = np.reshape(state, [1, 1])

            if reward == -10:
                penalties += 1

            epochs += 1

        test_total_penalties += penalties
        test_total_epochs += epochs

    print("**********************************")
    print("Results")
    print("**********************************")
    print("Epochs per episode: {}".format(test_total_epochs / test_num_of_episodes))
    print("Penalties per episode: {}".format(test_total_penalties / test_num_of_episodes))
    # fixme model is walking on the field for too long and makes a lot of wrong pickups/drops
