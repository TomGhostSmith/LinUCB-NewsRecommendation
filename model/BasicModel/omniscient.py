import math
import numpy
import random
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class Omniscient(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        self.rounds = 0
        self.armSelectCount = numpy.zeros(config.armCount)
        self.armReward = numpy.zeros(config.armCount)

        self.name = "Omniscient"

    def train(self, trainset:Dataset):
        for _, record in trainset.iterRecords():
            self.armSelectCount[record.arm] += 1
            self.armReward[record.arm] += record.reward
        
        self.armReward = [self.armReward[idx] / self.armSelectCount[idx] for idx in range(config.armCount)]


    def selectArm(self, context=None):
        # it is a common practice to try all arms at the beginning, so we won't ignore any unselected arms for their rewards are zero
        maxReward = max(self.armReward)
        maxRewardArms = [idx for idx in range(config.armCount) if self.armReward[idx] == maxReward]
        action = random.choice(maxRewardArms)
        thisReward = maxReward

        return action, self.armReward[action]
        # return action, thisReward

    def update(self, record:Record):
        self.rounds += 1


