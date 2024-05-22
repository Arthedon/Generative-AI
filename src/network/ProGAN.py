import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torch import optim

from loss.WGAN import WassersteinDiscriminatorLoss, WassersteinGeneratorLoss
from utils.logger import Logger
from utils.training_utils import LRScheduler
import os
from tqdm.auto import tqdm

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working with {device}...")
torch.set_default_device(device)

class PixelNorm(nn.Module):
    def __init__(self, epsilon = 1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        x = x / torch.sqrt((1 / x.size(1)) * torch.sum(x ** 2, dim = 1, keepdim = True) + self.epsilon)
        return x

class MiniBatchStdDev(nn.Module):
    def __init__(self, groupSize = 4):
        super(MiniBatchStdDev, self).__init__()
        self.groupSize = torch.tensor(groupSize)
    
    def forward(self, x):
        groupSize = torch.minimum(self.groupSize, torch.tensor(x.shape[0]))
        tensorShape = x.shape
        y = torch.reshape(x, [groupSize, -1, tensorShape[1], tensorShape[2], tensorShape[3]])
        y -= torch.mean(y, dim = 0, keepdim = True)
        y = torch.mean(torch.square(y), dim = 0)
        y = torch.mean(y, axis = [1, 2, 3], keepdim = True)
        y = torch.tile(y, [groupSize, 1, tensorShape[2], tensorShape[3]])
        return torch.concat([x, y], dim = 1)
    
class ProGANDiscriminator(nn.Module):
    def __init__(self, imageShape: tuple = (128, 128, 3), configuration: dict = {"channels": [128, 256, 512, 1024]}) -> None:
        super().__init__()
        height, width, self.inputChannel = imageShape[0], imageShape[1], imageShape[2]
        inputFeaturesSize = imageShape[0] * imageShape[1]
        self.configuration = configuration

        convolutionalBlocks = []
        inputChannel = configuration["channels"][0]
        for idx, outputChannel in enumerate(configuration["channels"]):
            resolution = 2 ** (len(configuration["channels"]) - (idx + 1) + 2)
            if idx != len(configuration["channels"]) - 1:
                convolutionalBlock = nn.ModuleDict({
                                                    "fromRGB": nn.Sequential(nn.Conv2d(in_channels = self.inputChannel, out_channels = inputChannel,
                                                                                       kernel_size = 1,
                                                                                       stride = 1,
                                                                                       padding = 0, device = device),
                                                                             PixelNorm().to(device),
                                                                             nn.LeakyReLU(negative_slope = 0.2)),
                                                    "conv0": nn.Conv2d(in_channels = inputChannel, out_channels = outputChannel,
                                                                       kernel_size = 3,
                                                                       stride = 1,
                                                                       padding = 1, device = device),
                                                    "norm0": PixelNorm().to(device),
                                                    "act0": nn.LeakyReLU(negative_slope = 0.2),
                                                    "conv1": nn.Conv2d(in_channels = outputChannel, out_channels = outputChannel,
                                                                       kernel_size = 3,
                                                                       stride = 1,
                                                                       padding = 1, device = device),
                                                    "norm1": PixelNorm().to(device),
                                                    "act1": nn.LeakyReLU(negative_slope = 0.2),
                                                    "downscale": nn.AvgPool2d(kernel_size = 2, stride = 2)})
            else:
                convolutionalBlock = nn.ModuleDict({
                                                    "fromRGB": nn.Sequential(nn.Conv2d(in_channels = self.inputChannel, out_channels = inputChannel,
                                                                                       kernel_size = 1,
                                                                                       stride = 1,
                                                                                       padding = 0, device = device),
                                                                             PixelNorm().to(device),
                                                                             nn.LeakyReLU(negative_slope = 0.2)),
                                                    "miniBatchStdDev": MiniBatchStdDev(),
                                                    "conv0": nn.Conv2d(in_channels = inputChannel + 1, out_channels = outputChannel,
                                                                       kernel_size = 3,
                                                                       stride = 1,
                                                                       padding = 1, device = device),
                                                    "norm0": PixelNorm().to(device),
                                                    "act0": nn.LeakyReLU(negative_slope = 0.2),
                                                    "conv1": nn.Conv2d(in_channels = outputChannel, out_channels = outputChannel,
                                                                       kernel_size = 4,
                                                                       stride = 1,
                                                                       padding = 0, device = device),
                                                    "norm1": PixelNorm().to(device),
                                                    "act1": nn.LeakyReLU(negative_slope = 0.2)})
            
            convolutionalBlocks.append((f"{resolution}x{resolution}", convolutionalBlock))
            inputChannel = outputChannel

        self.convolutionalBlocks = nn.ModuleDict(dict((key, value) for key, value in convolutionalBlocks))

        inputFeaturesSize = inputChannel
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = inputFeaturesSize, out_features = 1)
        )

    def forward(self, x: torch.Tensor, resolution: int = -1) -> torch.Tensor:
        mustStart, imageFromRGB = False, True
        if resolution == -1:
            resolution = 2 ** (len(self.configuration["channels"]) + 2)
        for imageSize, block in self.convolutionalBlocks.items():
            if int(imageSize.split("x")[0]) == resolution:
                mustStart = True
            if not mustStart:
                continue
            for name, layer in block.items():
                if name != "fromRGB" or imageFromRGB:
                    x = layer(x)
                    imageFromRGB = False
        x = self.output(x)
        return x

class ProGANGenerator(nn.Module):
    def __init__(self, imageShape: tuple = (128, 128, 3), configuration: dict = {"latentSpaceSize": 100, "channels": [1024, 512, 256, 128]}) -> None:
        super().__init__()
        self.inputChannel = configuration["latentSpaceSize"]
        height, width, self.outputChannel = imageShape[0], imageShape[1], imageShape[2]
        self.imageShape = imageShape
        self.configuration = configuration

        convolutionalBlocks = []
        inputChannel = self.inputChannel
        for idx, outputChannel in enumerate(configuration["channels"]):
            resolution = 2 ** (idx + 2)
            if idx == 0:
                convolutionalBlock = nn.ModuleDict({
                                                    "dense": nn.Linear(in_features = inputChannel, out_features = outputChannel * 4 * 4),
                                                    "reshape": nn.Unflatten(dim = 1, unflattened_size = (-1, 4, 4)),
                                                    "act0": nn.LeakyReLU(negative_slope = 0.2),
                                                    "conv0": nn.Conv2d(in_channels = outputChannel, out_channels = outputChannel,
                                                                        kernel_size = 3,
                                                                        stride = 1,
                                                                        padding = 1, device = device),
                                                    "norm0": PixelNorm().to(device),
                                                    "act1": nn.LeakyReLU(negative_slope = 0.2),
                                                    "toRGB": nn.Conv2d(in_channels = outputChannel, out_channels = self.outputChannel,
                                                                kernel_size = 1,
                                                                stride = 1,
                                                                padding = 0, device = device)})
            else:
                kernelSize = 3 if idx == len(configuration['channels']) - 1 else 1
                padding = 1 if idx == len(configuration['channels']) - 1 else 0
                convolutionalBlock = nn.ModuleDict({
                                                    "upscale": nn.UpsamplingNearest2d(scale_factor = 2),
                                                    "conv0": nn.Conv2d(in_channels = inputChannel, out_channels = outputChannel,
                                                                        kernel_size = 3,
                                                                        stride = 1,
                                                                        padding = 1, device = device),
                                                    "norm0": PixelNorm().to(device),
                                                    "act0": nn.LeakyReLU(negative_slope = 0.2),
                                                    "conv1": nn.Conv2d(in_channels = outputChannel, out_channels = outputChannel,
                                                                        kernel_size = 3,
                                                                        stride = 1,
                                                                        padding = 1, device = device),
                                                    "norm1": PixelNorm().to(device),
                                                    "act1": nn.LeakyReLU(negative_slope = 0.2),
                                                    "toRGB": nn.Conv2d(in_channels = outputChannel, out_channels = self.outputChannel,
                                                                        kernel_size = kernelSize,
                                                                        stride = 1,
                                                                        padding = padding, device = device)})
            
            convolutionalBlocks.append((f"{resolution}x{resolution}", convolutionalBlock))
            inputChannel = outputChannel
        
        self.convolutionalBlocks = nn.ModuleDict(dict((key, value) for key, value in convolutionalBlocks))
        
    def forward(self, x: torch.Tensor, resolution: int = -1) -> torch.Tensor:
        mustExit = False
        if resolution == -1:
            resolution = 2 ** (len(self.configuration["channels"]) + 2)
        for imageSize, block in self.convolutionalBlocks.items():
            if int(imageSize.split("x")[0]) == resolution:
                mustExit = True
            for name, layer in block.items():
                if name != "toRGB" or mustExit:
                    x = layer(x)
            if mustExit:
                break
        return x

class ProGAN():
    def __init__(self, imageShape: tuple,
                 configuration: dict = {"Discriminator": {"channels": [128, 256, 256, 512, 512]}, # useless in this code
                                        "Generator": {"latentSpaceSize": 100, "channels": [512, 512, 256, 256, 128]},
                                        "Training": {"extraSteps": 1, "penaltyRatio": 10, "epsilonDrift": 1e-3,
                                                     "epochUpscaling": [40, 80, 120, 160, 200], "epochAdaptation": 10},
                                                     "batchSizes": {"128": 64, "64": 128, "32": 256, "32": 512, "16": 1024, "4": 2048}}) -> None:
        super(ProGAN, self).__init__()
        self.imageShape = imageShape
        self.configuration = configuration
        architecture = configuration["Generator"]["channels"]
        configuration["Discriminator"]["channels"] = list(reversed(architecture))
        self.discriminatorNet = ProGANDiscriminator(self.imageShape, configuration["Discriminator"])
        self.generatorNet = ProGANGenerator(self.imageShape, configuration["Generator"])
        self.discriminatorLossCriterion = WassersteinDiscriminatorLoss()
        self.generatorLossCriterion = WassersteinGeneratorLoss()
        self.trainingConfiguration = configuration["Training"]


    def sample(self, sampleSize: int) -> torch.Tensor:
        z = torch.randn(size = (sampleSize, self.configuration["Generator"]["latentSpaceSize"])).to(device)
        return z
    
    def generator_forward(self, sampleSize: int, resolution: int = -1) -> torch.Tensor:
        z = self.sample(sampleSize)
        imageTensor = self.generatorNet(z, resolution)
        return imageTensor
    
    def discriminator_forward(self, images: torch.Tensor, resolution: int = -1) -> torch.Tensor:
        x = self.discriminatorNet(images, resolution)
        return x
    
    def merge_images(self, images: torch.Tensor, batchSize: int, resolution: int = -1):
        fakeImages = self.generator_forward(batchSize, resolution)
        alpha = torch.rand((batchSize, 1, 1, 1)).half().to(device)
        meanImages = alpha * images + (1 - alpha) * fakeImages
        yPred = self.discriminator_forward(meanImages, resolution).squeeze()
        return meanImages, yPred

    def gradient_penalty(self, meanImages: torch.Tensor, yPred: torch.Tensor):
        ones = torch.ones_like(yPred)
        gradParameters = torch.autograd.grad(outputs = yPred, inputs = meanImages, create_graph = True, grad_outputs = ones)[0]
        gradParameters = gradParameters.reshape(gradParameters.shape[0], -1)
        gradientNorm = gradParameters.norm(2, dim = 1)
        gradientPenalty = torch.mean((gradientNorm - 1) ** 2)
        return gradientPenalty

    def train_discriminator(self, optimizer, realData: torch.Tensor,  currentResolution: int, adaptationConfig: tuple):
        self.discriminatorNet.zero_grad()
        
        batchSize = realData.size(0)
        noise = self.sample(batchSize)
        fakeData = self.generatorNet(noise, currentResolution)

        criticsRealData = self.discriminatorNet(realData, currentResolution).view(-1)
        criticsFakeData = self.discriminatorNet(fakeData.detach(), currentResolution).view(-1)
        
        if adaptationConfig[0]:
            transform = transforms.Resize((currentResolution // 2, currentResolution // 2))
            adaptationCriticsRealData = self.discriminatorNet(transform(realData), currentResolution // 2).view(-1)
            adaptativeFakeData = self.generatorNet(noise, currentResolution // 2)
            adaptationCriticsFakeData = self.discriminatorNet(adaptativeFakeData.detach(), currentResolution // 2).view(-1)
            batchLoss = adaptationConfig[1] * self.discriminatorLossCriterion(adaptationCriticsRealData, adaptationCriticsFakeData) + \
                        (1 - adaptationConfig[1]) * self.discriminatorLossCriterion(criticsRealData, criticsFakeData)
        else:
            batchLoss = self.discriminatorLossCriterion(criticsRealData, criticsFakeData)
        
        mergedImages, yPred = self.merge_images(realData, batchSize, currentResolution)
        lossPenalty = self.gradient_penalty(mergedImages, yPred)
        totalBatchLoss = batchLoss + lossPenalty * self.trainingConfiguration["penaltyRatio"] + \
                         torch.square(criticsRealData).mean() * self.trainingConfiguration["epsilonDrift"] 

        totalBatchLoss.backward()
        optimizer.step()
        
        return batchLoss
        
    def train_generator(self, optimizer, batchSize: int, currentResolution: int, adaptationConfig: tuple):
        self.generatorNet.train()
        self.generatorNet.zero_grad()

        noise = self.sample(batchSize)
        fakeData = self.generatorNet(noise, currentResolution)
        
        criticsFakeData = self.discriminatorNet(fakeData, currentResolution).view(-1)

        if adaptationConfig[0]:
            adaptativeFakeData = self.generatorNet(noise, currentResolution // 2)
            adaptationCriticsFakeData = self.discriminatorNet(adaptativeFakeData.detach(), currentResolution // 2).view(-1)
            batchLoss = adaptationConfig[1] * self.generatorLossCriterion(adaptationCriticsFakeData) + \
                        (1 - adaptationConfig[1]) * self.generatorLossCriterion(criticsFakeData)
        else:
            batchLoss = self.generatorLossCriterion(criticsFakeData)

        batchLoss.backward()
        optimizer.step()
        
        return batchLoss
    
    def train_gan(self, numEpochs = 1e2, learningRate = 2e-4, optimizer: tuple = None, datasetPath: str = "../data/", 
                  continueTraining: bool = False, currentResolution: int = 4, currentEpoch: int = 0):
        
        progressBar = tqdm(range(int(numEpochs)))
        logger = Logger(model_name = 'ProWGAN', data_name = 'DOGS')
        
        batchSizes = self.trainingConfiguration["batchSizes"]
        preprocessing = transforms.Compose([
                            transforms.Resize((currentResolution, currentResolution)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transforms.Lambda(lambda x: x.to(device))])
        trainDataset = ds.ImageFolder(root = datasetPath, transform = preprocessing)      
        trainLoader = DataLoader(trainDataset, batch_size = batchSizes[str(currentResolution)],
                                 shuffle = True, generator = torch.Generator(device))

        weightsDirectoryPath = "../weights/ProWGAN/"
        GeneratorPath, DiscriminatorPath = "generator/", "discriminator/"
        bestWeightsPath = "best_model_weights"
        modelWeightsPath = "GAN_weights_"
        finalWeightsPath = "final_weights"
        weightsExtension = ".pth"

        if not os.path.isdir(weightsDirectoryPath + GeneratorPath):
            os.mkdir(weightsDirectoryPath + GeneratorPath)
        if not os.path.isdir(weightsDirectoryPath + DiscriminatorPath):
            os.mkdir(weightsDirectoryPath + DiscriminatorPath)


        if continueTraining:
            self.discriminatorNet.load_state_dict(torch.load(weightsDirectoryPath + DiscriminatorPath + finalWeightsPath + weightsExtension))
            self.generatorNet.load_state_dict(torch.load(weightsDirectoryPath + GeneratorPath + finalWeightsPath + weightsExtension))
        
        if not optimizer:
            discriminatorOptimizer = optim.Adam(self.discriminatorNet.parameters(), lr = learningRate, betas = (0, 0.99))
            generatorOptimizer = optim.Adam(self.generatorNet.parameters(), lr = learningRate, betas = (0, 0.99))
        else:
            discriminatorOptimizer, generatorOptimizer = optimizer

        # self.scheduler = LRScheduler(decayEpochs = 3 * numEpochs // 4,
        #                              optimizers = (generatorOptimizer, discriminatorOptimizer), 
        #                              minLearningRate = self.configuration["Training"]["minLearningRate"])

        discriminatorTrainingLoss, generatorTrainingLoss = [], []

        noise = self.sample(16)
        bestGeneratorLoss = float("inf")
        adaptationMode, transitionRatio = False, 0
        os.system('clear')
        for epoch in progressBar:
            
            if epoch + currentEpoch in self.trainingConfiguration["epochUpscaling"]:
                currentResolution *= 2

                preprocessing = transforms.Compose([
                            transforms.Resize((currentResolution, currentResolution)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transforms.Lambda(lambda x: x.to(device))])

                trainDataset = ds.ImageFolder(root = datasetPath, transform = preprocessing)      

                trainLoader = DataLoader(trainDataset, batch_size = batchSizes[str(currentResolution)],
                                         shuffle = True, generator = torch.Generator(device))
                
                adaptationMode, epochStart = True, (epoch + currentEpoch)
            
            if adaptationMode:
                transitionRatio = 1 - (epoch + currentEpoch - epochStart) / self.trainingConfiguration["epochAdaptation"]
                if transitionRatio == 0: adaptationMode = False
            
            discriminatorBatchLoss, generatorBatchLoss, generatorLoss = 0.0, 0.0, 0.0
            for idx, (realData, _) in enumerate(trainLoader):
                if idx % 10 == 0:
                    print(f"[{epoch}/{numEpochs}] Batch number {idx}/{int(len(trainDataset) / batchSizes[str(currentResolution)])}, Loss - discriminator: " + \
                          f"{discriminatorBatchLoss:.5f}, generator: {generatorBatchLoss:.5f}", end = '\r')

                for _ in range(self.trainingConfiguration["extraSteps"]):
                    imageNoise = torch.randn_like(realData) * 0.1
                    realData += imageNoise
                    realData = torch.clamp(realData, 0, 1)
                    discriminatorBatchLoss = self.train_discriminator(discriminatorOptimizer, realData, currentResolution, (adaptationMode, transitionRatio))

                generatorBatchLoss = self.train_generator(generatorOptimizer, batchSizes[str(currentResolution)], currentResolution, (adaptationMode, transitionRatio))
                generatorLoss += generatorBatchLoss
                
                generatorTrainingLoss.append(generatorBatchLoss)
                discriminatorTrainingLoss.append(discriminatorBatchLoss)
                
                if idx % 500 == 0:
                    testImages = self.generatorNet(noise, currentResolution).detach().cpu()
                    logger.log_images(
                        testImages, 16, 
                        epoch, idx, len(trainLoader)
                    )
            
            # self.scheduler.step(epoch)

            if generatorLoss < bestGeneratorLoss:
                bestGeneratorLoss = generatorLoss
                torch.save(self.generatorNet.state_dict(), weightsDirectoryPath + GeneratorPath + bestWeightsPath + weightsExtension)
                torch.save(self.discriminatorNet.state_dict(), weightsDirectoryPath + DiscriminatorPath + bestWeightsPath + weightsExtension)

            if epoch % 10 == 0:
                torch.save(self.discriminatorNet.state_dict(), weightsDirectoryPath + DiscriminatorPath + modelWeightsPath + str(epoch) + weightsExtension)
                torch.save(self.generatorNet.state_dict(), weightsDirectoryPath + GeneratorPath + modelWeightsPath + str(epoch) + weightsExtension)

            os.system('clear')

            progressBar.set_description(f'Discriminator Loss: {discriminatorBatchLoss:.3e}, Generator Loss: {generatorBatchLoss:.3e},'\
                                        + f' Current Resolution : {currentResolution}x{currentResolution}')

        torch.save(self.discriminatorNet.state_dict(), weightsDirectoryPath + DiscriminatorPath + finalWeightsPath + weightsExtension)
        torch.save(self.generatorNet.state_dict(), weightsDirectoryPath + GeneratorPath + finalWeightsPath + weightsExtension)

        return discriminatorTrainingLoss, generatorTrainingLoss
