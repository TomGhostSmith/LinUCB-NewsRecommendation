from model.dataset import Dataset
from model.record import Record
class MAB():
    def __init__(self, args) -> None:
        pass

    def train(self, trainset:Dataset):
        pass

    def selectArm(self, context=None)-> tuple:  # return (index of arm, regret for this round)
        pass

    def update(self, record:Record):
        pass