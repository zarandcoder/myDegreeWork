import math

from gym.utils import seeding
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow.compat.v1 as tf
import sys
import cv2
import numpy as np
import random
import gym
from gym import spaces
from PIL import Image
from matplotlib import pyplot as plt


MICROSNAP = 1
PLACE = 2
SOFT_GREEN = 3
SOFTEST_GREEN = 4
BGR_COLORS = {
    1: (255, 175, 0),  # Car Blue
    2: (0, 255, 0),  # Place Green
    3: (0, 125, 0),  # Place Soft Green
    4: (0, 75, 0)  # Place Softest Green
}

# Hyperparameters
GRID_SIZE = 20
ALPHA = 0.0001  # Learning rate 0.5...0.001
GAMMA = 0.9998  # Discount factor 0.9...0.999
MAX_EPSILON = 0.99
MIN_EPSILON = 0.05
LAMBDA = 0.001
BATCH_SIZE = 64
MEMORY_CAPACITY = 200_000
UPDATE_TARGET_FREQUENCY = 1000
MAX_DISTANCE_TO_PLACE = 2
EPOCHS = 50

TOTAL_REWARD_LIST = []


class MicroSnapCar:
    """
    MicroSnap instance class
    """
    def __init__(self, size_of_env):
        self.size = size_of_env
        self.x = np.random.randint(0, size_of_env)
        self.y = np.random.randint(0, size_of_env)

    def __str__(self):
        return f"x: ({self.x}, y: {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class CityLocation:
    """
    CityLocation instance class
    """
    def __init__(self, place_name, x=0, y=0):
        self.place_name = place_name
        self.x = x
        self.y = y


# Follow the gym environment guidelines
class BarcelonaEnv(gym.Env):
    """
        Description:
            Grid environment for Barcelona Scenario with six places with revenue generation.
            Microsnap car is moving inside this environment. The car can move in any direction to the bounds.
            The goal is to find places with big revenues an generate profit over the whole week.
            The revenue depends on the weekday and daytime, so the car should move on gained
            experience at certain day and time to the right place.

        Observation:
            Num     Observation             Min         Max
            0       Car Position (x)        0           19
            1       Car Positon (y)         0           19
            2       Weekday                 0 (Monday)  6 (Sunday)
            3       Daytime (4 h Slot)      0 (6-10)    3 (18-22)

        Actions:
            Num     Action
            0       Move to the right
            1       Move to the left
            2       Move up
            3       Move down
            4       Stay on place

        Reward:
            Generally - reward of the agent depends on the location, day and daytime.
            If agent places himself on the one of six places then reward is positive
            else -1 for every step taken.

        Starting State:
            All observation values are assigned to random values

        Episode Termination:
            Agent took 50 steps in the Grid Map.
            That means the car need ca. 4.2 minutes to come over one square in the map
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    cur_epsilon = 0

    def __init__(self):
        super(BarcelonaEnv, self).__init__()

        self.actions_per_timeslot = 50
        self.timeslots = {
            0: '06:00 - 10:00',
            1: '10:00 - 14:00',
            2: '14:00 - 18:00',
            3: '18:00 - 22:00'
        }
        self.days = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }

        self.episode_step = 0
        self.reward = 0

        # Highest and Lowest observation spaces
        high = np.array([19, 19, 6, 3], dtype=np.int32)
        low = np.array([0, 0, 0, 0], dtype=np.int32)

        self.action_space = spaces.Discrete(5)
        # TODO Check really the right parameters
        self.observation_space = spaces.Box(low=low, high=high, shape=None)
        self.seed()

        self.state = None

        self.viewer = None
        self.done = False

        # Generate Locations
        self.loc1 = CityLocation("Stadion", 2, 2)
        self.loc2 = CityLocation("Ausflug", 2, 12)
        self.loc3 = CityLocation("Plaza", 13, 17)
        self.loc4 = CityLocation("El Raval", 16, 15)
        self.loc5 = CityLocation("links Nebenschauplatz", 6, 10)
        self.loc6 = CityLocation("rechts Nebenschauplatz", 6, 17)

        self.microsnap = MicroSnapCar(size_of_env=GRID_SIZE)

        self.cur_day = 0
        self.cur_timeslot = 0
        self.cur_actionstep = 0

    def reset(self):
        self.episode_step = 0
        self.microsnap.x = x = self.np_random.randint(0, 19)
        self.microsnap.y = y = self.np_random.randint(0, 19)
        _day = self.cur_day
        _timeslot = self.cur_timeslot
        self.state = x, y, _day, _timeslot
        return np.array(self.state)

    def step(self, action):
        self.episode_step += 1
        x, y, _day, _time = self.state

        if action == 0:
            x, y = self.move(x=1, y=0)
        if action == 1:
            x, y = self.move(x=-1, y=0)

        if action == 2:
            x, y = self.move(x=0, y=1)

        if action == 3:
            x, y = self.move(x=0, y=-1)

        if action == 4:
            x, y = self.move(x=0, y=0)

        reward = self.compute_reward(self.state)
        done = self.done
        info = {}

        self.state = x, y, _day, _time

        return self.state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    def render(self, mode='human'):
        if mode == 'human':
            img = self.get_image()
            img = img.resize((500, 500))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontColor = (255, 175, 0)
            lineType = 2

            img = np.array(img)

            cv2.putText(img=img, text=f"Day -> {self.days[self.cur_day]}", org=(10, 435), fontFace=font,
                        fontScale=fontScale, color=fontColor, lineType=lineType)
            cv2.putText(img=img, text=f"Time -> {self.timeslots[self.cur_timeslot]}", org=(10, 450), fontFace=font,
                        fontScale=fontScale, color=fontColor, lineType=lineType)
            cv2.putText(img=img, text=f"Step Nr -> {self.cur_actionstep}", org=(10, 465), fontFace=font,
                        fontScale=fontScale, color=fontColor, lineType=lineType)
            cv2.putText(img=img, text=f"x -> {self.microsnap.x}", org=(10, 480),fontFace=font,
                        fontScale=fontScale,color=fontColor,lineType=lineType)
            cv2.putText(img=img, text=f"y -> {self.microsnap.y}", org=(10, 495), fontFace=font,
                        fontScale=fontScale, color=fontColor, lineType=lineType)

            eps = '{:.3f}'.format(self.cur_epsilon)
            cv2.putText(img=img, text=f"Epsilon -> {eps}", org=(10, 15), fontFace=font,
                        fontScale=fontScale, color=fontColor, lineType=lineType)

            cv2.imshow("Barcelona", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        else:
            super(BarcelonaEnv, self).render(mode=mode)  # just raise an exception

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def move(self, x=False, y=False):
        """
        Move inside the Grid World and check if out of bounds
        :return x,y coordinates after move
        """
        # If no value for x, move randomly
        if not x:
            self.microsnap.x += np.random.randint(-1, 2)
        else:
            self.microsnap.x += x

        # If no value for y, move randomly
        if not y:
            self.microsnap.y += np.random.randint(-1, 2)
        else:
            self.microsnap.y += y

        # make sure we don't go out of bounds
        if self.microsnap.x < 0:
            self.microsnap.x = 0
        elif self.microsnap.x > GRID_SIZE - 1:
            self.microsnap.x = GRID_SIZE - 1
        if self.microsnap.y < 0:
            self.microsnap.y = 0
        elif self.microsnap.y > GRID_SIZE - 1:
            self.microsnap.y = GRID_SIZE - 1

        return self.microsnap.x, self.microsnap.y

    def is_not_weekend(self, day):
        return day < 4

    def compute_reward(self, observation):
        """
            Calculate the gained reward per step taken.
            Check also current day and time
            :param observation
            :return reward taken after one step
        """
        locations_cords = [(1, 2), (1, 12), (13, 18), (16, 15), (6, 10), (6, 17)]
        # Places with rewards per timeslot
        rewards_weekday = {
            # Optimal behaviour reward = 400
            locations_cords[0]: [10, 10, 10, 10],
            locations_cords[1]: [100, 100, 20, 10],  # Optimum: 1) 100, 2) 100
            locations_cords[2]: [10, 20, 20, 20],
            locations_cords[3]: [10, 10, 100, 100],  # Optimum: 3) 100, 4) 100
            locations_cords[4]: [30, 30, 30, 30],
            locations_cords[5]: [40, 40, 40, 40]
        }

        rewards_weekend = {
            # Optimal behaviour reward = 400
            locations_cords[0]: [100, 100, 10, 20],  # Optimum: 1) 100, 2) 100
            locations_cords[1]: [10, 20, 20, 10],
            locations_cords[2]: [20, 20, 100, 50],  # Optimum: 3) 100
            locations_cords[3]: [25, 25, 20, 100],  # Optimum: 4) 100
            locations_cords[4]: [10, 10, 10, 10],
            locations_cords[5]: [10, 10, 10, 10]
        }

        for location in locations_cords:
            reward_curr_place = 0
            distance_place_car = np.abs(observation[0] - location[0]) + np.abs(observation[1] - location[1])

            # calculate reward if car is near enough to the current place
            if distance_place_car < MAX_DISTANCE_TO_PLACE:
                # add reward relative to the distance between place and car
                if self.is_not_weekend(observation[2]):
                    reward_curr_place = \
                        int((1 - distance_place_car / (MAX_DISTANCE_TO_PLACE + 1)) * rewards_weekday[location][3])
                else:
                    reward_curr_place = \
                        int((1 - distance_place_car / (MAX_DISTANCE_TO_PLACE + 1)) * rewards_weekend[location][3])

                return reward_curr_place

        reward = -1
        return reward

    def get_image(self):
        """
            Get current position of the car and places as image
            :return: image
        """
        env = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        env[self.microsnap.x][self.microsnap.y] = BGR_COLORS[MICROSNAP]

        # Show soft green points for nearby revenue
        env[self.loc1.x][self.loc1.y] = BGR_COLORS[PLACE]
        env[self.loc1.x+1][self.loc1.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x-1][self.loc1.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x][self.loc1.y+1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x][self.loc1.y-1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x+1][self.loc1.y+1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x-1][self.loc1.y-1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x+1][self.loc1.y-1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x-1][self.loc1.y+1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc1.x + 2][self.loc1.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc1.x - 2][self.loc1.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc1.x][self.loc1.y + 2] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc1.x][self.loc1.y - 2] = BGR_COLORS[SOFTEST_GREEN]

        env[self.loc2.x][self.loc2.y] = BGR_COLORS[PLACE]
        env[self.loc2.x + 1][self.loc2.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x - 1][self.loc2.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x][self.loc2.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x][self.loc2.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x + 1][self.loc2.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x - 1][self.loc2.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x + 1][self.loc2.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x - 1][self.loc2.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc2.x + 2][self.loc2.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc2.x - 2][self.loc2.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc2.x][self.loc2.y + 2] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc2.x][self.loc2.y - 2] = BGR_COLORS[SOFTEST_GREEN]

        env[self.loc3.x][self.loc3.y] = BGR_COLORS[PLACE]
        env[self.loc3.x + 1][self.loc3.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x - 1][self.loc3.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x][self.loc3.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x][self.loc3.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x + 1][self.loc3.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x - 1][self.loc3.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x + 1][self.loc3.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x - 1][self.loc3.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc3.x + 2][self.loc3.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc3.x - 2][self.loc3.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc3.x][self.loc3.y + 2] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc3.x][self.loc3.y - 2] = BGR_COLORS[SOFTEST_GREEN]

        env[self.loc4.x][self.loc4.y] = BGR_COLORS[PLACE]
        env[self.loc4.x + 1][self.loc4.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x - 1][self.loc4.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x][self.loc4.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x][self.loc4.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x + 1][self.loc4.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x - 1][self.loc4.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x + 1][self.loc4.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x - 1][self.loc4.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc4.x + 2][self.loc4.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc4.x - 2][self.loc4.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc4.x][self.loc4.y + 2] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc4.x][self.loc4.y - 2] = BGR_COLORS[SOFTEST_GREEN]

        env[self.loc5.x][self.loc5.y] = BGR_COLORS[PLACE]
        env[self.loc5.x + 1][self.loc5.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x - 1][self.loc5.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x][self.loc5.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x][self.loc5.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x + 1][self.loc5.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x - 1][self.loc5.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x + 1][self.loc5.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x - 1][self.loc5.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc5.x + 2][self.loc5.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc5.x - 2][self.loc5.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc5.x][self.loc5.y + 2] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc5.x][self.loc5.y - 2] = BGR_COLORS[SOFTEST_GREEN]

        env[self.loc6.x][self.loc6.y] = BGR_COLORS[PLACE]
        env[self.loc6.x + 1][self.loc6.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x - 1][self.loc6.y] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x][self.loc6.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x][self.loc6.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x + 1][self.loc6.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x - 1][self.loc6.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x + 1][self.loc6.y - 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x - 1][self.loc6.y + 1] = BGR_COLORS[SOFT_GREEN]
        env[self.loc6.x + 2][self.loc6.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc6.x - 2][self.loc6.y] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc6.x][self.loc6.y + 2] = BGR_COLORS[SOFTEST_GREEN]
        env[self.loc6.x][self.loc6.y - 2] = BGR_COLORS[SOFTEST_GREEN]

        img = Image.fromarray(env, 'RGB')

        return img


class Brain:
    """
        The Brain class encapsulates the neural network.
        The network consists of one hidden layer
        of 64 neurons, with ReLU activation function.
        The final layer will consist of five neurons,
        one for each available action. Their activation function
        will be softmax. Remember that we are trying to approximate the Q function,
        which in essence can be of any real value. Therefore we can’t restrict the output
        from the network and the softmax activation works well.
    """
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

        self.model = self._create_model()
        self.model_ = self._create_model()

    def _create_model(self):
        """
            The three most common loss functions are:
                - 'binary_crossentropy' for binary classification.
                - 'sparse_categorical_crossentropy' for multi -class classification.
                - 'mse' (mean squared error) for regression.
        """
        model = Sequential()

        layer1 = Dense(output_dim=8, activation='relu', input_dim=self.state_space)
        layer2 = Dense(output_dim=self.action_space, activation='softmax')

        model.add(layer=layer1)
        model.add(layer=layer2)

        # Compile the model
        opt = Adam(learning_rate=ALPHA)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

        # Textual summary of the model
        model.summary()
        return model

    def train(self, x, y, epochs=1, verbose=0):
        """
            performs supervised training step with batch
        """
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, state, target=False):
        """
            Predicts the Q function values in state s
        """
        if target:
            return self.model_.predict(state)
        else:
            return self.model.predict(state)

    def predict_one(self, state, target=False):
        if type(state) == tuple:
            __state = np.array(state)
            return self.predict(__state.reshape(1, self.state_space), target=target).flatten()
        return self.predict(state.reshape(1, self.state_space), target=target).flatten()

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())


class Memory:
    """
        The purpose of the Memory class is to store experience.
        It almost feels superfluous in the current problem,
        but we will implement it anyway. It is a good abstraction
        for the experience replay part and will allow us to easily
        upgrade it to more sophisticated algorithms later on.

        The add(sample) method stores the experience into the internal array,
        making sure that it does not exceed its capacity.
        The other method sample(n) returns n random samples from the memory.
    """
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        """
            Add sample to memory
        """
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        """
            Returns random batch of n samples
        """
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def is_full(self):
        return len(self.samples) >= self.capacity


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, action_cnt):
        self.action_cnt = action_cnt

    def act(self, s):
        return random.randint(0, self.action_cnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    # Experience replay
    def replay(self):
        pass


class Agent:
    """
        The Agent class acts as a container for the agent related properties and methods.
        The act(s) method implements the ε-greedy policy. With probability epsilon,
        it chooses a random action, otherwise it selects the best action the current ANN returns.
    """
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

        self.brain = Brain(state_space=state_space, action_space=action_space)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, state):
        """
            Decides what action to take in state
            :param state: current state
            :return: random move action or argmax from Q-Table
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space-1)
        else:
            return np.argmax(self.brain.predict_one(state=state))

    def observe(self, sample):
        """
            adds sample (s, a, r, s_) to memory
        """
        self.memory.add(sample=sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.update_target_model()

        # Debug Q function
        if self.steps % 100 == 0:
            S = np.array([1, 5, 1, 1])
            pred = agent.brain.predict_one(S)
            # print(pred[0])
            sys.stdout.flush()

        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        BarcelonaEnv.cur_epsilon = self.epsilon
        print(f'Current epsilon: {self.epsilon}')

    def replay(self):
        """
            replays memories and improves
        """
        batch = self.memory.sample(BATCH_SIZE)
        batch_len = len(batch)

        no_state = np.zeros(self.state_space)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(state=states)
        p_ = self.brain.predict(state=states_, target=True)

        x = np.zeros((batch_len, self.state_space))
        y = np.zeros((batch_len, self.action_space))

        for i in range(batch_len):
            o = batch[i]
            state = o[0]
            action = o[1]
            reward = o[2]
            state_ = o[3]

            t = p[i]

            if state_ is None:
                t[action] = reward
            else:
                t[action] = reward + GAMMA * np.amax(p_[i])

            x[i] = state
            y[i] = t

        # TODO: How many epochs are needed?
        self.brain.train(x=x, y=y, epochs=EPOCHS, verbose=2)


class EnvironmentWrapper:
    """
    EnvironmentWrapper class
    """
    TOTAL_REWARD = 0

    def __init__(self):
        self.env = BarcelonaEnv()

    def run(self, agent, mode=False):
        """
        :param mode: false - random agent, true - agent
        :param agent:
        :return: Total reward per timeslot
        """
        s = self.env.reset()
        R = 0

        # Iterate over one episode (50 timesteps)
        for t in range(self.env.actions_per_timeslot):
            self.env.cur_actionstep = t
            if mode:
                self.env.render()
                # pass
            a = agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            R += r  # Add step reward to timeslot total reward
            print(f'Step: {t} | Current reward: {r} | Total reward per timeslot: {R}')

        if mode:
            self.TOTAL_REWARD += R


# Another kind of loss function
# Alternative to MSE
HUBER_LOSS_DELTA = 1.0
def huber_loss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


if __name__ == '__main__':
    state = None
    env = EnvironmentWrapper()
    state_count = env.env.observation_space.shape[0]
    action_count = env.env.action_space.n

    agent = Agent(state_space=state_count, action_space=action_count)
    random_agent = RandomAgent(action_cnt=action_count)

    try:
        while not random_agent.memory.is_full():
            env.run(agent=random_agent)

        agent.memory.samples = random_agent.memory.samples
        random_agent = None

        # Day -> Timeslot -> Max action
        for d in env.env.days:
            env.env.cur_day = d

            for t in env.env.timeslots:
                env.env.cur_timeslot = t
                env.run(agent=agent, mode=True)

            TOTAL_REWARD_LIST.append(env.TOTAL_REWARD)

            if d == 6:
                # Data for plotting
                t = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                s = TOTAL_REWARD_LIST
                fig, ax = plt.subplots()
                ax.plot(t, s)

                ax.set(xlabel='Day', ylabel='Reward', title='Total Reward')
                ax.grid(which='both')
                plt.show()
        print(f' Total reward {env.TOTAL_REWARD}')
    finally:
        agent.brain.model.save("models/microsnap-dqn.h5")
        print('Model saved')

