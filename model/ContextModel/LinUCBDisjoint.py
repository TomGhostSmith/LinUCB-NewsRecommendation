import numpy
import random
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class LinUCBDisjoint(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        self.alpha = args['alpha']
        self.method = args['method']
        self.useUser = args['useUser']
        self.slice = slice(0, config.armContextDimension) if self.useUser else slice(0, config.articleContextDimension)
        self.size = config.armContextDimension if self.useUser else config.articleContextDimension

        self.rounds = 0
        self.armSelectCount = numpy.zeros(config.armCount)
        self.armReward = numpy.zeros(config.armCount)

        self.A = numpy.array([numpy.identity(self.size)] * config.armCount)
        self.b = numpy.zeros((config.armCount, self.size))

        self.name = r"LinUCB Disjoint ($\alpha$={:.2f}, info={}, method={})".format(self.alpha, "Article + User" if self.useUser else "Article only", self.method)

    def train(self, trainset:Dataset):
        pass

    def preprocessContext(self, feature):
        feature = feature[self.slice]
        if (self.method == 'normalize'):
            if (not numpy.all(feature == 0)):
                feature = feature /  numpy.linalg.norm(feature)  # normalize / standarize / min-max normalization
        elif (self.method == 'standarize'):
            if (max(feature) != min(feature)):
                feature = (feature - numpy.mean(feature)) / numpy.var(feature)
            else:
                feature = numpy.zeros_like(feature)
        elif (self.method == 'one'):
            if (max(feature) != min(feature)):
                feature = (feature - min(feature)) / (max(feature) - min(feature))
            else:
                feature = numpy.zeros_like(feature)
        
        return feature


    def selectArm(self, context=None):
        p = [0] * config.armCount
        for i in range(config.armCount):
            feature = self.preprocessContext(context[i])
            theta = numpy.linalg.inv(self.A[i]).dot(self.b[i])
            p[i] = theta.T.dot(feature) + self.alpha * numpy.sqrt(feature.T.dot(numpy.linalg.inv(self.A[i])).dot(feature))
        
        maxReward = max(p)
        maxRewardArms = [idx for idx in range(config.armCount) if p[idx] == maxReward]
        action = random.choice(maxRewardArms)

        return action, self.armReward[action]
        # return action, thisReward

    def update(self, record:Record):
        self.rounds += 1
        arm = record.arm
        reward = record.reward

        feature = self.preprocessContext(record.context[arm])

        self.A[arm] += feature.dot(feature.T)
        self.b[arm] += reward * feature

        self.armReward[arm] = (self.armReward[arm] * self.armSelectCount[arm] + reward) / (self.armSelectCount[arm] + 1)
        self.armSelectCount[arm] += 1


