import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
from keras.callbacks import TensorBoard


class ModifiedTensorBoard(TensorBoard):
    MODEL_NAME = '2x256'  # -> Neuron Layers and batch size
    """
    TensorBoard provides the visualization and
    tooling needed for machine learning experimentation
    at localhost:8888
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)


    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


def plot_rewards(episode_rewards, rival_rewards, show_every, fn=""):
    moving_avg = np.convolve(episode_rewards, np.ones((show_every,)) / show_every, mode='valid')
    moving_avg_rv = np.convolve(rival_rewards, np.ones((show_every,)) / show_every, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg, 'b')
    plt.plot([i for i in range(len(moving_avg_rv))], moving_avg_rv, 'r')
    plt.ylabel(f"Reward {show_every}ma")
    plt.xlabel("episode #")

    if len(fn) > 0:
        plt.savefig(fn)
    else:
        plt.show()

    plt.close()
    return