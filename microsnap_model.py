import numpy as np


class MicroSnapCar:
    def __init__(self, size_of_env):
        self.size = size_of_env
        self.x = np.random.randint(0, size_of_env)
        self.y = np.random.randint(0, size_of_env)

    def __str__(self):
        return f"x: ({self.x}, y: {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Location:
    def __init__(self, place_name, x=0, y=0):
        self.place_name = place_name
        self.x = x
        self.y = y