import numpy
import random
import sklearn
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class WarmLinUCBDisjoint(MAB):
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

        self.A = numpy.array([numpy.identity(self.size + 1)] * config.armCount)
        self.b = numpy.zeros((config.armCount, self.size + 1))

        self.name = r"Warm LinUCB Disjoint ($\alpha$={:.2f}, info={}, method={})".format(self.alpha, "Article + User" if self.useUser else "Article only", self.method)

    def train(self, trainset:Dataset):
        contexts = numpy.zeros((len(trainset.records), self.size + 1))
        rewards = numpy.zeros(len(trainset.records))
        for idx, record in trainset.iterRecords():
            selectedArm = record.arm
            contexts[idx] = self.preprocessContext(record.context[selectedArm])
            rewards[idx] = record.reward
        
        self.warmModel = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        self.warmModel.fit(contexts, rewards)

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
        feature = feature.tolist()
        feature.append(1)

        return numpy.array(feature)


    def selectArm(self, context=None):
        p = [0] * config.armCount
        for i in range(config.armCount):
            feature = self.preprocessContext(context[i])
            theta = numpy.linalg.inv(self.A[i]).dot(self.b[i])
            warmth = self.warmModel.predict_log_proba([feature])[0][1]
            p[i] = theta.T.dot(feature) + self.alpha * numpy.sqrt(feature.T.dot(numpy.linalg.inv(self.A[i])).dot(feature)) + warmth
        
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

        x = feature.reshape((self.size + 1, 1))

        self.A[arm] += x.dot(x.T)
        self.b[arm] += (reward * x).squeeze(1)

        self.armReward[arm] = (self.armReward[arm] * self.armSelectCount[arm] + reward) / (self.armSelectCount[arm] + 1)
        self.armSelectCount[arm] += 1


