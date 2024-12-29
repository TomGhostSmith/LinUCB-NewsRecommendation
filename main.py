from model.BasicModel.epsilonGreedy import EpsilonGreedy
from model.BasicModel.UCB import UCB
from model.BasicModel.randomModel import RandomModel
from model.BasicModel.omniscient import Omniscient
from model.BasicModel.ThompsonSampling import ThompsonSampling

from model.ContextModel.LinUCBDisjoint import LinUCBDisjoint
from model.ContextModel.LinUCBHybrid import LinUCBHybrid
from model.ContextModel.LinUCBHybridExt import LinUCBHybridExt

from model.WarmStartModel.WarmUCB import WarmUCB
from model.WarmStartModel.WarmEpsilonGreedy import WarmEpsilonGreedy
from model.WarmStartModel.WarmLinUCBHybrid import WarmLinUCBHybrid

from model.evaluator import Evaluator
from model.dataset import Dataset

import random

def compareModels(models, dataset:Dataset, testName):
    trainset, testset = dataset.split(0.2, seed=2024)
    for model in models:
        model.train(trainset)
    evaluators = [Evaluator(model) for model in models]
    for evaluator in evaluators:
        evaluator.evaluate(testset)
        # evaluator.draw()
    
    Evaluator.drawImage(evaluators, testName)


def main():
    random.seed(2024)
    models = list()
    # models.append(EpsilonGreedy({"epsilon": 0}))
    # models.append(EpsilonGreedy({"epsilon": 0.05}))
    # models.append(EpsilonGreedy({"epsilon": 0.1}))
    # models.append(WarmEpsilonGreedy({"epsilon": 0.05, "warmType": "user", "method": "normalize"}))
    # models.append(WarmEpsilonGreedy({"epsilon": 0.05, "warmType": "article", "method": "normalize"}))
    # models.append(WarmEpsilonGreedy({"epsilon": 0.05, "warmType": "user*article", "method": "normalize"}))
    # models.append(WarmEpsilonGreedy({"epsilon": 0.05, "warmType": "user+article", "method": "normalize"}))
    # models.append(WarmEpsilonGreedy({"epsilon": 0.05, "warmType": "user+article", "method": "standarize"}))
    # models.append(WarmEpsilonGreedy({"epsilon": 0.05, "warmType": "user+article", "method": "one"}))
    models.append(UCB({"alpha": 0}))
    models.append(UCB({"alpha": 0.05}))
    models.append(UCB({"alpha": 0.1}))
    # models.append(WarmUCB({"alpha": 0.05, "warmType": "user", "method": "normalize"}))
    # models.append(WarmUCB({"alpha": 0.05, "warmType": "article", "method": "normalize"}))
    # models.append(WarmUCB({"alpha": 0.05, "warmType": "user*article", "method": "normalize"}))
    # models.append(WarmUCB({"alpha": 0.05, "warmType": "user+article", "method": "normalize"}))
    # models.append(WarmUCB({"alpha": 0.05, "warmType": "user+article", "method": "standarize"}))
    # models.append(WarmUCB({"alpha": 0.05, "warmType": "user+article", "method": "one"}))
    models.append(RandomModel({}))
    models.append(Omniscient({}))
    # models.append(ThompsonSampling({"alpha": 1, "beta": 1}))
    # models.append(LinUCBDisjoint({"alpha": 0.05, "useUser": True, "method": "normalize"}))
    # models.append(LinUCBDisjoint({"alpha": 0.05, "useUser": True, "method": "standarize"}))
    # models.append(LinUCBDisjoint({"alpha": 0.05, "useUser": True, "method": "one"}))
    # models.append(LinUCBDisjoint({"alpha": 0.1, "useUser": True, "method": "normalize"}))
    # models.append(LinUCBDisjoint({"alpha": 0.05, "useUser": False, "method": "normalize"}))
    # models.append(LinUCBDisjoint({"alpha": 0.05, "useUser": False, "method": "standarize"}))
    # models.append(LinUCBDisjoint({"alpha": 0.05, "useUser": False, "method": "one"}))
    # models.append(LinUCBDisjoint({"alpha": 0.1, "useUser": False, "method": "normalize"}))
    models.append(LinUCBDisjoint({"alpha": 0.2, "useUser": False, "method": "normalize"}))
    models.append(LinUCBDisjoint({"alpha": 0.2, "useUser": True, "method": "normalize"}))
    # models.append(LinUCBHybrid({"alpha": 0.05, "method": "normalize"}))
    models.append(LinUCBHybrid({"alpha": 0.2, "method": "normalize"}))
    # models.append(WarmLinUCBHybrid({"alpha": 0.05, "warmType": "article", "method": "normalize"}))
    # models.append(WarmLinUCBHybrid({"alpha": 0.05, "warmType": "user", "method": "normalize"}))
    # models.append(WarmLinUCBHybrid({"alpha": 0.05, "warmType": "user+article", "method": "normalize"}))
    # models.append(WarmLinUCBHybrid({"alpha": 0.05, "warmType": "user*article", "method": "normalize"}))
    # models.append(LinUCBHybridExt({"alpha": 0.05, "method": "normalize"}))
    models.append(LinUCBHybridExt({"alpha": 0.2, "method": "normalize"}))
    dataset = Dataset.fromFile("dataset/dataset.txt")
    compareModels(models, dataset, "test")


if (__name__ == '__main__'):
    main()