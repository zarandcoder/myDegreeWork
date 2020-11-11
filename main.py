from microsnap_drl.dqn_agent import DQNAgent
from microsnap_drl.dqn_agent import env
from microsnap_drl.utils import ModifiedTensorBoard

from tqdm import tqdm
import numpy as np
import time


# Constants and parameters
MODEL_NAME = '2x256'  # -> Neuron Layers and batch size
EPISODES = 20000  # How many times should agent learn?
SHOW_PREVIEW = True  # True if agent action should be visible
AGGREGATE_STATES_EVERY = 10  # After how much Episodes to see window with action

MIN_REWARD = -200

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
TIMESLOTS_PER_DAY = 4

def main():
    epsilon = 1
    tensor_board = ModifiedTensorBoard(log_dir="logs/{}-{}".format(ModifiedTensorBoard.MODEL_NAME, int(time.time())))
    agent = DQNAgent(tensor_board)

    # tqdm is for visualization of progress bar
    #for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
    for episode in range(0, EPISODES):
        tensor_board.step = episode

        episode_reward = 0
        step = 1

        current_state = env.reset()




        done = False
        while not done:
            # np.random.random generates a double in range 0 < x < 1
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            new_state, reward, done = env.step(action)

            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATES_EVERY:
                env.render()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        '''
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATES_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATES_EVERY:])\
                             /len(ep_rewards[-AGGREGATE_STATES_EVERY:])

            min_reward = min(ep_rewards[-AGGREGATE_STATES_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATES_EVERY:])

            tensor_board.update_stats(reward_avg=average_reward,
                                           reward_min=min_reward,
                                           reward_max=max_reward,
                                           epsilon=epsilon)

            if min_reward >= MIN_REWARD:
                agent.model.save(f'old_models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        '''

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


if __name__ == '__main__':
    pass