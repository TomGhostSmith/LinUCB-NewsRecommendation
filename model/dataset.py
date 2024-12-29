import random
from model.record import Record
class Dataset():
    def __init__(self) -> None:
        self.records = list()
        # self.index = 0  # the index to read next time
    
    @classmethod
    def fromFile(clz, filePath):
        data = clz()
        with open(filePath) as fp:
            data.records = [Record(line) for line in fp]
        
        return data

    def split(self, trainsetproportion, seed):
        random.seed(seed)
        records = self.records.copy()
        random.shuffle(records)
        splitIndex = round(len(records) * trainsetproportion)
        trainset = Dataset()
        trainset.records = records[:splitIndex]
        testset = Dataset()
        testset.records = records[splitIndex:]

        return trainset, testset
    
    def iterRecords(self, startIndex=0):
        records = self.records[startIndex:] + self.records[:startIndex]
        indexs = list(range(startIndex, len(self.records))) + list(range(0, startIndex))
        return zip(indexs, records)
    
    # def getNextRecord(self, arm=None):
    #     if (arm is not None):
    #         # strategy: find the next record with the corresponding action
    #         for index in range(self.index, len(self.records)):
    #             if self.records[index].arm == arm:
    #                 self.index = index
    #                 return self.records[index]
    #         for index in range(0, self.index):
    #             if self.records[index].arm == arm:
    #                 self.index = index
    #                 return self.records[index]
            
    #     # if there is no record with the same action, or it is the first time to get record
    #     record = self.records[self.index]
    #     self.index += 1
    #     return record
