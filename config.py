class Config():
    def __init__(self) -> None:
        self.armCount = 10
        self.armContextDimension = 10
        self.evaluateRounds = 1000

        self.articleContextDimension = 5
        self.userContextDimension = self.armContextDimension - self.articleContextDimension

        self.articelContextIndex = slice(0, self.articleContextDimension)
        self.userContextIndex = slice(self.articleContextDimension, self.armContextDimension)

        self.seed = 2024

config = Config()