import numpy
from config import config

class Record():
    def __init__(self, line) -> None:
        terms = [int(t) for t in line.strip().split(' ')]
        self.arm = terms[0] - 1   # origin arm is 1 to 10
        self.reward = terms[1]
        self.context = numpy.zeros((config.armCount, config.armContextDimension))
        for idx in range(config.armCount):
            self.context[idx] = terms[idx*config.armContextDimension + 2 : (idx+1)*config.armContextDimension + 2]