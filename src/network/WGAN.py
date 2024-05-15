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

class WGANDiscriminator(nn.Module):
    def __init__(self, imageShape: tuple = (128, 128, 3), configuration: dict = {"channels": [16, 64, 256]}) -> None:
        super().__init__()
        height, width = imageShape[0], imageShape[1]
        inputFeaturesSize = imageShape[0] * imageShape[1]
        self.convolutionalLayers = nn.Sequential()
        inputChannel = imageShape[2]
        for idx, outputChannel in enumerate(configuration["channels"]):
            self.convolutionalLayers.append(nn.Conv2d(in_channels = inputChannel, out_channels = outputChannel, 
                                                        kernel_size = 3,
                                                        stride = 2, 
                                                        padding = 1, device = device))
            height //= 2
            width //= 2
            inputFeaturesSize /= 4
            if idx != 0:
                self.convolutionalLayers.append(nn.LayerNorm(normalized_shape = [outputChannel, height, width], device = device))
            self.convolutionalLayers.append(nn.LeakyReLU(0.2, inplace = True))
            self.convolutionalLayers.append(nn.Dropout(0.3))
            inputChannel = outputChannel

        inputFeaturesSize *= outputChannel
        self.output = nn.Sequential(
            nn.Conv2d(in_channels = inputChannel, out_channels = 1, kernel_size = 3, stride = 2, padding = 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutionalLayers(x)
        x = self.output(x)
        return x

class WGANGenerator(nn.Module):
    def __init__(self, imageShape: tuple = (128, 128, 3), configuration: dict = {"latentSpaceSize": 100, "channels": [1024, 512, 256, 128]}) -> None:
        super().__init__()
        self.inputChannel = configuration["latentSpaceSize"]
        self.outputFeatures = imageShape[0] * imageShape[1] * imageShape[2]
        self.imageShape = imageShape

        self.configuration = configuration

        self.convolutionalLayers = nn.Sequential()

        inputChannel = self.inputChannel
        for idx, outputChannel in enumerate(configuration["channels"]):
            if idx == 0:
                self.convolutionalLayers.append(nn.ConvTranspose2d(in_channels = inputChannel, out_channels = outputChannel,
                                                                kernel_size = 4,
                                                                stride = 1,
                                                                padding = 0, device = device))
                self.convolutionalLayers.append(nn.BatchNorm2d(outputChannel).to(device))
                self.convolutionalLayers.append(nn.ReLU(True))
                inputChannel = outputChannel
            else:
                self.convolutionalLayers.append(nn.ConvTranspose2d(in_channels = inputChannel, out_channels = outputChannel,
                                                                kernel_size = 4,
                                                                stride = 2,
                                                                padding = 1, device = device))
                self.convolutionalLayers.append(nn.BatchNorm2d(outputChannel).to(device))
                self.convolutionalLayers.append(nn.ReLU(True))
                inputChannel = outputChannel

        self.convolutionalLayers.append(nn.ConvTranspose2d(in_channels = inputChannel, out_channels = 3,
                                                        kernel_size = 4,
                                                        stride = 2,
                                                        padding = 1, device = device))
        self.convolutionalLayers.append(nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutionalLayers(x)
        return x
    
class WGAN():
    def __init__(self, imageShape: tuple,
                 configuration: dict = {"Discriminator": {"channels": [16, 64, 256]},
                                        "Generator": {"latentSpaceSize": 100, "channels": [1024, 512, 256, 128]},
                                        "Training": {"extraSteps": 5, "penaltyRatio": 10}}) -> None:
        super(WGAN, self).__init__()
        self.imageShape = imageShape
        self.configuration = configuration
        self.discriminatorNet = WGANDiscriminator(self.imageShape, self.configuration["Discriminator"])
        self.generatorNet = WGANGenerator(self.imageShape, self.configuration["Generator"])
        self.discriminatorLossCriterion = WassersteinDiscriminatorLoss()
        self.generatorLossCriterion = WassersteinGeneratorLoss()
        self.penaltyRatio = configuration["Training"]["penaltyRatio"]
        self.extraSteps = configuration["Training"]["extraSteps"]

    def sample(self, sampleSize: int) -> torch.Tensor:
        z = torch.randn(size = (sampleSize, self.configuration["Generator"]["latentSpaceSize"], 1, 1)).to(device)
        return z
    
    def generator_forward(self, sampleSize: int) -> torch.Tensor:
        z = self.sample(sampleSize)
        imageTensor = self.generatorNet(z)
        return imageTensor
    
    def discriminator_forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.discriminatorNet(images)
        return x
    
    def merge_images(self, images: torch.Tensor, batchSize: int):
        fakeImages = self.generator_forward(batchSize)
        alpha = torch.rand((batchSize, 1, 1, 1)).half().to(device)
        meanImages = alpha * images + (1 - alpha) * fakeImages
        yPred = self.discriminator_forward(meanImages).squeeze()
        return meanImages, yPred

    def gradient_penalty(self, meanImages: torch.Tensor, yPred: torch.Tensor):
        ones = torch.ones_like(yPred)
        gradParameters = torch.autograd.grad(outputs = yPred, inputs = meanImages, create_graph = True, grad_outputs = ones)[0]
        gradParameters = gradParameters.reshape(gradParameters.shape[0], -1)
        gradientNorm = gradParameters.norm(2, dim = 1)
        gradientPenalty = torch.mean((gradientNorm - 1) ** 2)
        return gradientPenalty

    def train_discriminator(self, optimizer, realData: torch.Tensor):
        self.discriminatorNet.zero_grad()
        
        batchSize = realData.size(0)
        noise = self.sample(batchSize)
        fakeData = self.generatorNet(noise)

        criticsRealData = self.discriminatorNet(realData).view(-1)
        criticsFakeData = self.discriminatorNet(fakeData.detach()).view(-1)
        
        batchLoss = self.discriminatorLossCriterion(criticsRealData, criticsFakeData)
        mergedImages, yPred = self.merge_images(realData, batchSize)
        lossPenalty = self.gradient_penalty(mergedImages, yPred)
        totalBatchLoss = batchLoss + lossPenalty * self.penaltyRatio

        totalBatchLoss.backward()
        optimizer.step()
        
        return batchLoss
        
    def train_generator(self, optimizer, batchSize: int):
        self.generatorNet.train()
        self.generatorNet.zero_grad()

        noise = self.sample(batchSize)
        fakeData = self.generatorNet(noise)
        
        criticsFakeData = self.discriminatorNet(fakeData).view(-1)        
        batchLoss = self.generatorLossCriterion(criticsFakeData)

        batchLoss.backward()
        optimizer.step()
        
        return batchLoss
    
    def train_gan(self, numEpochs = 1e2, learningRate = 2e-4, optimizer: tuple = None, batchSize = 64, datasetPath: str = "../data/", 
                  continueTraining: bool = False):
        
        progressBar = tqdm(range(int(numEpochs)))
        logger = Logger(model_name = 'WGAN', data_name = 'DOGS')

        preprocessing = transforms.Compose([
                               transforms.Resize(self.imageShape[0]),
                               transforms.CenterCrop(self.imageShape[0]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transforms.Lambda(lambda x: x.to(device))])


        trainDataset = ds.ImageFolder(root = datasetPath,
                                      transform = preprocessing)
        
        # Create the dataloader
        trainLoader = DataLoader(trainDataset, batch_size = batchSize,
                                 shuffle = True, generator = torch.Generator(device))

        weightsDirectoryPath = "../weights/WGAN/"
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
            discriminatorOptimizer = optim.Adam(self.discriminatorNet.parameters(), lr = learningRate, betas = (0, 0.9))
            generatorOptimizer = optim.Adam(self.generatorNet.parameters(), lr = learningRate, betas = (0, 0.9))
        else:
            discriminatorOptimizer, generatorOptimizer = optimizer

        # self.scheduler = LRScheduler(decayEpochs = 3 * numEpochs // 4,
        #                              optimizers = (generatorOptimizer, discriminatorOptimizer), 
        #                              minLearningRate = self.configuration["Training"]["minLearningRate"])

        discriminatorTrainingLoss, generatorTrainingLoss = [], []

        noise = self.sample(16)
        bestGeneratorLoss = float("inf")
        os.system('clear')
        for epoch in progressBar:
            discriminatorBatchLoss, generatorBatchLoss, generatorLoss = 0.0, 0.0, 0.0
            for idx, (realData, _) in enumerate(trainLoader):
                if idx % 10 == 0:
                    print(f"[{epoch}/{numEpochs}] Batch number {idx}/{int(len(trainDataset) / batchSize)}, Loss - discriminator: " + \
                          f"{discriminatorBatchLoss:.5f}, generator: {generatorBatchLoss:.5f}", end = '\r')

                for _ in range(self.extraSteps):
                    imageNoise = torch.randn_like(realData) * 0.1
                    realData += imageNoise
                    realData = torch.clamp(realData, 0, 1)
                    discriminatorBatchLoss = self.train_discriminator(discriminatorOptimizer, realData)

                generatorBatchLoss = self.train_generator(generatorOptimizer, batchSize)
                generatorLoss += generatorBatchLoss
                
                generatorTrainingLoss.append(generatorBatchLoss)
                discriminatorTrainingLoss.append(discriminatorBatchLoss)
                
                if idx % 500 == 0:
                    testImages = self.generatorNet(noise).detach().cpu()
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

            progressBar.set_description(f'Discriminator Loss: {discriminatorBatchLoss:.3e}, Generator Loss: {generatorBatchLoss:.3e}')

        torch.save(self.discriminatorNet.state_dict(), weightsDirectoryPath + DiscriminatorPath + finalWeightsPath + weightsExtension)
        torch.save(self.generatorNet.state_dict(), weightsDirectoryPath + GeneratorPath + finalWeightsPath + weightsExtension)

        return discriminatorTrainingLoss, generatorTrainingLoss
