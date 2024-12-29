import math
import numpy
import random
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class RandomModel(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        self.rounds = 0
        self.armSelectCount = numpy.zeros(config.armCount)
        self.armReward = numpy.zeros(config.armCount)

        self.name = "Random"

    def train(self, trainset:Dataset):
        pass

    def selectArm(self, context=None):
        # it is a common practice to try all arms at the beginning, so we won't ignore any unselected arms for their rewards are zero
        unselectedArms = [idx for idx in range(config.armCount) if self.armSelectCount[idx] == 0]
        if len(unselectedArms) > 0:
            action = random.choice(unselectedArms)
            thisReward = 0
            maxReward = 1
        else:
            maxReward = 1
            action = random.choice(range(config.armCount))
            thisReward = maxReward


        return action, self.armReward[action]
        # return action, thisReward

    def update(self, record:Record):
        self.rounds += 1
        arm = record.arm
        reward = record.reward
        self.armReward[arm] = (self.armReward[arm] * self.armSelectCount[arm] + reward) / (self.armSelectCount[arm] + 1)
        self.armSelectCount[arm] += 1


