import numpy
from Utils import IOUtils
from model.MAB import MAB
from model.dataset import Dataset
from config import config
import matplotlib.pyplot as plt

class Evaluator():
    def __init__(self, model:MAB) -> None:
        self.model:MAB = model
        self.selectedArms = [0] * config.evaluateRounds
        self.realRewards = [0] * config.evaluateRounds
        self.recordedRegrets = [0] * config.evaluateRounds
    
    @classmethod
    def drawImage(clz, evaluators, imageName):
        plt.figure(figsize=(20, 12))
        evaluatorList = [(evaluator, evaluator.realRewards[-1]) for evaluator in evaluators]
        evaluatorList = sorted(evaluatorList, key=lambda a: a[1], reverse=True)
        for evaluator, score in evaluatorList:
            plt.plot(range(config.evaluateRounds), evaluator.realRewards, label=f"{evaluator.model.name}: Reward={score:.4f}")
        plt.xlabel('round')
        plt.ylabel('reward')
        plt.legend()
        plt.savefig(f"img/{imageName}-reward.png")
        plt.close()

        plt.figure(figsize=(20, 12))
        evaluatorList = [(evaluator, evaluator.recordedRegrets[-1]) for evaluator in evaluators]
        evaluatorList = sorted(evaluatorList, key=lambda a: a[1])
        for evaluator, score in evaluatorList:
            plt.plot(list(range(config.evaluateRounds)), evaluator.recordedRegrets, label=f"{evaluator.model.name}: Regret={score:.4f}")
        plt.xlabel('round')
        plt.ylabel('regret')
        plt.legend()
        plt.savefig(f"img/{imageName}-regret.png")
        plt.close()
    
    def evaluate(self, dataset:Dataset):
        IOUtils.showInfo(f"Evaluating {self.model.name}")
        # lastArm = None

        # record = dataset.getNextRecord()
        recordIndex = 0
        totalReward = 0
        totalRegret = 0
        for round in range(config.evaluateRounds):
            foundRecord = False
            for index, record in dataset.iterRecords(recordIndex):
                selectedArm, predictedReward = self.model.selectArm(record.context)
                if (selectedArm == record.arm):
                    recordIndex = index + 1
                    foundRecord = True
                    self.model.update(record)
                    break
            
            if (foundRecord):
                totalReward += record.reward
            else:  # there is no record has corresponding arm selection, so we assign reward = 0
                totalReward += 0
            totalRegret += predictedReward - record.reward
            if (totalRegret > round+1):
                print("?")
            self.realRewards[round] = totalReward / (round + 1)
            self.recordedRegrets[round] = totalRegret / (round + 1)
            self.selectedArms[round] = selectedArm
        
        return self.realRewards, self.recordedRegrets

    def draw(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(config.evaluateRounds), self.realRewards, label=self.model.name)
        plt.xlabel('round')
        plt.ylabel('reward')
        plt.legend()
        plt.savefig(f"img/{self.model.name}-reward.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(list(range(config.evaluateRounds)), self.recordedRegrets, label=self.model.name)
        plt.xlabel('round')
        plt.ylabel('regret')
        plt.legend()
        plt.savefig(f"img/{self.model.name}-regret.png")
        plt.close()