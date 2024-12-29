import math
import numpy
import random
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class UCB(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        self.alpha = args['alpha']

        self.rounds = 0
        self.armSelectCount = numpy.zeros(config.armCount)
        self.armReward = numpy.zeros(config.armCount)

        self.name = r"UCB ($\alpha$={:.2f})".format(self.alpha)

    def train(self, trainset:Dataset):
        pass

    def selectArm(self, context=None):
        # it is a common practice to try all arms at the beginning, so we won't ignore any unselected arms for their rewards are zero
        unselectedArms = [idx for idx in range(config.armCount) if self.armSelectCount[idx] == 0]
        if len(unselectedArms) > 0:
            action = random.choice(unselectedArms)
            thisReward = 0
            if (self.rounds == 0):
                maxReward = 0
            else:
                maxReward = max([math.sqrt(self.alpha * (math.log(self.rounds) / self.armSelectCount[idx])) + self.armReward[idx] if self.armSelectCount[idx] > 0 else 0 for idx in range(config.armCount)])
        else:
            UCBValues = numpy.sqrt(self.alpha * (numpy.log(self.rounds) / self.armSelectCount)) + self.armReward
            maxReward = max(UCBValues)
            maxRewardArms = [idx for idx in range(config.armCount) if UCBValues[idx] == maxReward]
            action = random.choice(maxRewardArms)
            thisReward = maxReward



        # return action, maxReward
        return action, self.armReward[action]
        # return action, thisReward

    def update(self, record:Record):
        self.rounds += 1
        arm = record.arm
        reward = record.reward
        self.armReward[arm] = (self.armReward[arm] * self.armSelectCount[arm] + reward) / (self.armSelectCount[arm] + 1)
        self.armSelectCount[arm] += 1


