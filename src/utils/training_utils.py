class LRScheduler():
    def __init__(self, decayEpochs: int, optimizers: tuple, minLearningRate: float = 1e-7):
        super(LRScheduler, self).__init__()
        self.decayEpochs = decayEpochs
        self.minLearningRate = minLearningRate
        self.generatorOptimizer, self.discriminatorOptimizer = optimizers 
        self.generatorLearningRate = self.generatorOptimizer.param_groups[0]['lr']
        self.discriminatorLearningRate = self.discriminatorOptimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.decayEpochs:
            newGeneratorLearningRate = max(self.generatorLearningRate * (1 - (epoch / self.decayEpochs)), self.minLearningRate)
            self.generatorOptimizer.param_groups[0]['lr'] = newGeneratorLearningRate
            newDiscriminatorLearningRate = max(self.discriminatorLearningRate * (1 - (epoch / self.decayEpochs)), self.minLearningRate)
            self.discriminatorOptimizer.param_groups[0]['lr'] = newDiscriminatorLearningRate