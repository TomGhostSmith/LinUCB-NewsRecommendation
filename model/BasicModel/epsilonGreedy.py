import numpy
import random
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class EpsilonGreedy(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        self.epsilon = args['epsilon']

        self.rounds = 0
        self.armSelectCount = [0] * config.armCount
        self.armReward = [0] * config.armCount

        self.name = r"Epsilon Greedy ($\epsilon$={:.2f})".format(self.epsilon)

    def train(self, trainset:Dataset):
        pass

    def selectArm(self, context=None):
        # it is a common practice to try all arms at the beginning, so we won't ignore any unselected arms for their rewards are zero
        unselectedArms = [idx for idx in range(config.armCount) if self.armSelectCount[idx] == 0]
        maxReward = max(self.armReward)
        if len(unselectedArms) > 0:
            action = random.choice(unselectedArms)
            thisReward = 0
        elif (random.uniform(0, 1) < self.epsilon):
            action = random.choice(range(config.armCount))
            thisReward = self.armReward[action]
        else:
            maxRewardArms = [idx for idx in range(config.armCount) if self.armReward[idx] == maxReward]
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


