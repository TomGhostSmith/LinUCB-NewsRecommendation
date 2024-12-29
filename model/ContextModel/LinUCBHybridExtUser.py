import numpy
import random
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class LinUCBHybridExtUser(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        self.alpha = args['alpha']
        self.method = args['method']

        self.rounds = 0
        self.armSelectCount = numpy.zeros(config.armCount)
        self.armReward = numpy.zeros(config.armCount)

        # k: user/article combination
        # d: article dimension

        self.userContextDimension = (config.userContextDimension + 1) * (config.articleContextDimension + 1)  # assume that we can calculate outer of them

        self.A0 = numpy.identity(self.userContextDimension)
        self.b0 = numpy.zeros(self.userContextDimension)


        self.A = numpy.array([numpy.identity(config.userContextDimension + 1)] * config.armCount)
        self.B = numpy.zeros((config.armCount, config.userContextDimension + 1, self.userContextDimension))
        self.b = numpy.zeros((config.armCount, config.userContextDimension + 1))


        self.name = r"LinUCB Hybrid Ext. User ($\alpha$={:.2f})".format(self.alpha)

    def train(self, trainset:Dataset):
        pass

    def preprocessContext(self, feature):
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

        invA0 = numpy.linalg.inv(self.A0) 
        beta = invA0.dot(self.b0)
        for i in range(config.armCount):
            articleFeature = self.preprocessContext(context[i, config.articelContextIndex])
            userFeature = self.preprocessContext(context[i, config.userContextIndex])

            x = userFeature
            x.resize((config.userContextDimension + 1, 1))
            z = numpy.outer(userFeature, articleFeature)
            z.resize((self.userContextDimension, 1))
            invA = numpy.linalg.inv(self.A[i])
            B = self.B[i]
            b = self.b[i]

            theta = invA.dot(b - B.dot(beta))
            s = z.T.dot(invA0).dot(z) - 2 * z.T.dot(invA0).dot(B.T).dot(invA).dot(x) + x.T.dot(invA).dot(x) + x.T.dot(invA).dot(B).dot(invA0).dot(B.T).dot(invA).dot(x)
            p[i] = z.T.dot(beta) + x.T.dot(theta) + self.alpha * numpy.sqrt(s)
        
        maxReward = max(p)
        maxRewardArms = [idx for idx in range(config.armCount) if p[idx] == maxReward]
        action = random.choice(maxRewardArms)

        return action, self.armReward[action]
        # return action, thisReward

    def update(self, record:Record):
        self.rounds += 1
        arm = record.arm
        reward = record.reward

        articleFeature = self.preprocessContext(record.context[arm, config.articelContextIndex])
        userFeature = self.preprocessContext(record.context[arm, config.userContextIndex])

        x = userFeature
        x.resize((config.userContextDimension + 1, 1))
        z = numpy.outer(userFeature, articleFeature)
        z.resize((self.userContextDimension, 1))
        invA = numpy.linalg.inv(self.A[arm])
        B = self.B[arm]
        b = self.b[arm]

        self.A0 += B.T.dot(invA).dot(B)
        self.b0 += B.T.dot(invA).dot(b)
        self.A[arm] += x.dot(x.T)
        self.B[arm] += x.dot(z.T)
        self.b[arm] += (reward * x).squeeze(1)

        invA = numpy.linalg.inv(self.A[arm])
        B = self.B[arm]
        b = self.b[arm]

        self.A0 += z.dot(z.T) - B.T.dot(invA).dot(B)
        self.b0 += (reward * z).squeeze(1) - B.T.dot(invA).dot(b)


        self.armReward[arm] = (self.armReward[arm] * self.armSelectCount[arm] + reward) / (self.armSelectCount[arm] + 1)
        self.armSelectCount[arm] += 1


