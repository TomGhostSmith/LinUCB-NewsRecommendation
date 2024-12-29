import numpy
import random
import sklearn
from model.MAB import MAB
from model.dataset import Dataset
from model.record import Record
from config import config
class WarmEpsilonGreedy(MAB):
    def __init__(self, args) -> None:
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        self.epsilon = args['epsilon']
        self.warmType = args['warmType']
        self.method = args['method']

        self.rounds = 0
        self.armSelectCount = numpy.zeros(config.armCount)
        self.armReward = numpy.zeros(config.armCount)

        self.warmModel = None

        warmthDimension = {
            "user": config.userContextDimension + 1,
            "article": config.articleContextDimension + 1,
            "user+article": config.armContextDimension + 1,
            "user*article": (config.userContextDimension + 1) * (config.articleContextDimension + 1)
        }
        self.warmthDimention = warmthDimension[self.warmType]

        self.name = r"Epsilon Greedy ($\epsilon$={:.2f}, warm={}, method={})".format(self.epsilon, self.warmType, self.method)


    def train(self, trainset:Dataset):
        contexts = numpy.zeros((len(trainset.records), self.warmthDimention))
        rewards = numpy.zeros(len(trainset.records))
        for idx, record in trainset.iterRecords():
            selectedArm = record.arm
            contexts[idx] = self.getFeature(record.context[selectedArm])
            rewards[idx] = record.reward
        
        self.warmModel = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        self.warmModel.fit(contexts, rewards)

    
    def getFeature(self, feature):
        if (self.warmType == 'user'):
            feature = self.preprocessContext(feature[config.userContextIndex])
        elif (self.warmType == 'article'):
            feature = self.preprocessContext(feature[config.articelContextIndex])
        elif (self.warmType == 'user+article'):
            feature = self.preprocessContext(feature)
        elif (self.warmType == 'user*article'):
            userFeature = self.preprocessContext(feature[config.userContextIndex])
            articleFeature = self.preprocessContext(feature[config.articelContextIndex])
            feature = numpy.outer(userFeature, articleFeature)
            feature.resize((config.userContextDimension + 1) * (config.articleContextDimension + 1))
        return feature


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
        # it is a common practice to try all arms at the beginning, so we won't ignore any unselected arms for their rewards are zero
        unselectedArms = [idx for idx in range(config.armCount) if self.armSelectCount[idx] == 0]
        if len(unselectedArms) > 0:
            action = random.choice(unselectedArms)
            thisReward = 0
        elif (random.uniform(0, 1) < self.epsilon):
            action = random.choice(range(config.armCount))
            thisReward = self.armReward[action]
        else:
            warmths = numpy.zeros(config.armCount)
            for idx in range(config.armCount):
                feature = self.getFeature(context[idx])
                warmths[idx] = self.warmModel.predict_log_proba([feature])[0][1]
            EGValue = self.armReward + warmths
            maxReward = max(EGValue)
            maxRewardArms = [idx for idx in range(config.armCount) if EGValue[idx] == maxReward]
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


