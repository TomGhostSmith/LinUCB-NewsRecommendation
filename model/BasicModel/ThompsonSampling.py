import numpy
import random
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class ThompsonSampling(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        self.alpha = args['alpha']
        self.beta = args['beta']

        self.alphas = numpy.array([self.alpha] * config.armCount)
        self.betas = numpy.array([self.beta] * config.armCount)

        self.rounds = 0
        self.armSelectCount = [0] * config.armCount
        self.armReward = [0] * config.armCount

        self.name = r"Thompson Sampling ($\alpha$={:.2f}, $\beta$={:.2f})".format(self.alpha, self.beta)

    def train(self, trainset:Dataset):
        pass

    def selectArm(self, context=None):
        # it is a common practice to try all arms at the beginning, so we won't ignore any unselected arms for their rewards are zero
        unselectedArms = [idx for idx in range(config.armCount) if self.armSelectCount[idx] == 0]
        expectation = numpy.random.beta(self.alphas, self.betas)
        maxReward = max(expectation)
        if len(unselectedArms) > 0:
            action = random.choice(unselectedArms)
            thisReward = 0
        else:
            maxRewardArms = [idx for idx in range(config.armCount) if expectation[idx] == maxReward]
            action = random.choice(maxRewardArms)
            thisReward = maxReward

        return action, self.armReward[action]
        # return action, thisReward

    def update(self, record:Record):
        self.rounds += 1
        arm = record.arm
        reward = record.reward
        if (record.reward == 1):
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1
        self.armReward[arm] = (self.armReward[arm] * self.armSelectCount[arm] + reward) / (self.armSelectCount[arm] + 1)
        self.armSelectCount[arm] += 1


