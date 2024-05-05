import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torch import optim

from utils.logger import Logger
import os
from tqdm.auto import tqdm

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working with {device}...")
torch.set_default_device(device)

class GANDiscriminator(nn.Module):
    def __init__(self, imageShape: tuple = (512, 288, 3), architecture: dict = {"numClasses": 1, "channels": [16, 64, 256]}) -> None:
        super().__init__()
        self.inputChannel = imageShape[0] * imageShape[1]
        self.outputFeatures = architecture["numClasses"]

        self.convolutionnalLayers = nn.Sequential()
        inputChannel = imageShape[2]
        for idx, outputChannel in enumerate(architecture["channels"]):
            self.convolutionnalLayers.append(nn.Conv2d(in_channels = inputChannel, out_channels = outputChannel, 
                                                        kernel_size = 3,
                                                        stride = 2, 
                                                        padding = 1, device = device))
            if idx != 0:
                self.convolutionnalLayers.append(nn.BatchNorm2d(outputChannel, device = device))
            self.convolutionnalLayers.append(nn.LeakyReLU(0.2, inplace = True))
            self.convolutionnalLayers.append(nn.Dropout(0.3))
            inputChannel = outputChannel

        self.linearLayers = nn.Sequential()
        self.linearLayers.append(nn.Linear(in_features = int(self.inputChannel / (4 ** len(architecture["channels"])) * outputChannel), out_features = 1024,
                                           device = device))
        self.linearLayers.append(nn.Linear(in_features = 1024, out_features = self.outputFeatures, device = device))
        self.linearLayers.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutionnalLayers(x)
        x = x.view(x.size(0), -1)
        x = self.linearLayers(x)

        return x

class GANGenerator(nn.Module):
    def __init__(self, imageShape: tuple = (512, 288, 3), architecture: dict = {"latentSpaceSize": 100, "channels": [1024, 512, 256, 128]}) -> None:
        super().__init__()
        self.inputChannel = architecture["latentSpaceSize"]
        self.outputFeatures = imageShape[0] * imageShape[1] * imageShape[2]
        self.imageShape = imageShape

        self.architecture = architecture

        self.convolutionnalLayers = nn.Sequential()

        inputChannel = self.inputChannel
        for idx, outputChannel in enumerate(architecture["channels"]):
            if idx == 0:
                self.convolutionnalLayers.append(nn.ConvTranspose2d(in_channels = inputChannel, out_channels = outputChannel,
                                                                kernel_size = 4,
                                                                stride = 1,
                                                                padding = 0, device = device))
                self.convolutionnalLayers.append(nn.BatchNorm2d(outputChannel).to(device))
                self.convolutionnalLayers.append(nn.ReLU(True))
                inputChannel = outputChannel
            else:
                self.convolutionnalLayers.append(nn.ConvTranspose2d(in_channels = inputChannel, out_channels = outputChannel,
                                                                kernel_size = 4,
                                                                stride = 2,
                                                                padding = 1, device = device))
                self.convolutionnalLayers.append(nn.BatchNorm2d(outputChannel).to(device))
                self.convolutionnalLayers.append(nn.ReLU(True))
                inputChannel = outputChannel

        self.convolutionnalLayers.append(nn.ConvTranspose2d(in_channels = inputChannel, out_channels = 3,
                                                        kernel_size = 4,
                                                        stride = 2,
                                                        padding = 1, device = device))
        self.convolutionnalLayers.append(nn.Tanh())



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutionnalLayers(x)

        return x
    
class GAN():
    def __init__(self, imageShape: tuple, architecture: dict = {"Discriminator": {"numClasses": 1, "channels": [16, 64, 256]},
                                                                  "Generator": {"latentSpaceSize": 100, "channels": [1024, 512, 256, 128]}}) -> None:
        super().__init__()
        self.imageShape = imageShape
        self.architecture = architecture
        self.discriminatorNet = GANDiscriminator(self.imageShape, self.architecture["Discriminator"])
        self.generatorNet = GANGenerator(self.imageShape, self.architecture["Generator"])
        self.lossCriterion = nn.BCELoss()

    def sample(self, sampleSize: int) -> torch.Tensor:
        z = torch.randn(size = (sampleSize, self.architecture["Generator"]["latentSpaceSize"], 1, 1)).to(device)
        return z
    
    def generator_forward(self, sampleSize: int) -> torch.Tensor:
        z = self.sample(sampleSize)
        imageTensor = self.generatorNet(z)
        return imageTensor
    
    def discriminator_forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.discriminatorNet(images)
        return x
        
    def train_discriminator(self, optimizer, realData):
        # Train with real batch
        self.discriminatorNet.zero_grad()

        batchSize = realData.size(0)
        label = torch.full((batchSize,), 1, dtype = torch.float, device = device)

        predictions = self.discriminatorNet(realData).view(-1)

        errorRealData = self.lossCriterion(predictions, label)

        errorRealData.backward()
        batchAccuracy = predictions.mean().item()

        # Train with fake batch        
        noise = self.sample(batchSize)

        fakeData = self.generatorNet(noise)
        label.fill_(0)

        predictions = self.discriminatorNet(fakeData.detach()).view(-1)

        errorFakeData = self.lossCriterion(predictions, label)

        errorFakeData.backward()
        batchLoss = errorRealData + errorFakeData
        optimizer.step()
        
        return batchLoss, batchAccuracy, fakeData
        
    def train_generator(self, optimizer, fakeData):
        self.generatorNet.train()

        self.generatorNet.zero_grad()
        label = torch.full((fakeData.size(0),), 1, dtype = torch.float, device = device)

        predictions = self.discriminatorNet(fakeData).view(-1)
        
        batchLoss = self.lossCriterion(predictions, label)
        batchLoss.backward()
        batchAccuracy = predictions.mean().item()

        optimizer.step()
        
        return (batchLoss, batchAccuracy)
    
    def train_gan(self, numEpochs = 1e2, learningRate = 2e-4, optimizer: tuple = None, batchSize = 64, datasetPath: str = "../data/", 
                  continueTraining: bool = False):
        
        progressBar = tqdm(range(int(numEpochs)))
        logger = Logger(model_name = 'GAN', data_name = 'DOGS')

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

        weightsDirectoryPath = "../weights/GAN/"
        GeneratorPath, DiscriminatorPath = "generator/", "discriminator/"
        bestWeightsPath = "best_model_weights"
        modelWeightsPath = "GAN_weights_"
        finalWeightsPath = "final_weights"
        weightsExtension = ".pth"

        if continueTraining:
            self.discriminatorNet.load_state_dict(torch.load(weightsDirectoryPath + DiscriminatorPath + bestWeightsPath + weightsExtension))
            self.generatorNet.load_state_dict(torch.load(weightsDirectoryPath + GeneratorPath + bestWeightsPath + weightsExtension))
        if not optimizer:
            discriminatorOptimizer = optim.Adam(self.discriminatorNet.parameters(), lr = learningRate)
            generatorOptimizer = optim.Adam(self.generatorNet.parameters(), lr = learningRate)
        else:
            discriminatorOptimizer, generatorOptimizer = optimizer

        discriminatorTrainingLoss, generatorLoss, generatorAccuracy, discriminatorAccuracy = [], [], [], []
        bestGeneratorAccuracy = 0.0

        noise = self.sample(16)

        for epoch in progressBar:
            discriminatorBatchLoss, discriminatorBatchAccuracy, generatorBatchLoss, generatorBatchAccuracy = 0.0, 0.0, 0.0, 0.0
            meanDiscriminatorAccuracy, meanGeneratorAccuracy = 0.0, 0.0
            for idx, (realData, _) in enumerate(trainLoader):
                if idx % 10 == 0:
                    print(f"[{epoch}/{numEpochs}] Batch number {idx}/{int(len(trainDataset) / batchSize)}, Loss {discriminatorBatchLoss:.5f}, " + \
                          f"{generatorBatchLoss:.5f}, Accuracy: {meanDiscriminatorAccuracy / (idx + 1):.2f}, {meanGeneratorAccuracy / (idx + 1):.2f}",
                          end = '\r')
                discriminatorBatchLoss, discriminatorBatchAccuracy, fakeData = self.train_discriminator(discriminatorOptimizer, 
                                                                                                        realData)
                meanDiscriminatorAccuracy += discriminatorBatchAccuracy

                loss, accuracy = self.train_generator(generatorOptimizer,
                                                      fakeData)
                generatorBatchLoss = loss
                generatorBatchAccuracy = accuracy

                meanGeneratorAccuracy += generatorBatchAccuracy

                discriminatorTrainingLoss.append(discriminatorBatchLoss)
                generatorLoss.append(generatorBatchLoss)
                generatorAccuracy.append(generatorBatchAccuracy)
                
                if idx % 500 == 0:
                    testImages = self.generatorNet(noise).detach().cpu()
                    logger.log_images(
                        testImages, 16, 
                        epoch, idx, len(trainLoader)
                    )
            
            if generatorBatchAccuracy > bestGeneratorAccuracy:
                bestGeneratorAccuracy = generatorBatchAccuracy
                torch.save(self.generatorNet.state_dict(), weightsDirectoryPath + GeneratorPath + bestWeightsPath + weightsExtension)
                torch.save(self.discriminatorNet.state_dict(), weightsDirectoryPath + DiscriminatorPath + bestWeightsPath + weightsExtension)

            if epoch % 5 == 0:
                torch.save(self.discriminatorNet.state_dict(), weightsDirectoryPath + DiscriminatorPath + modelWeightsPath + str(epoch) + weightsExtension)
                torch.save(self.generatorNet.state_dict(), weightsDirectoryPath + GeneratorPath + modelWeightsPath + str(epoch) + weightsExtension)

            os.system('clear')

            progressBar.set_description(f'Discriminator Loss: {discriminatorBatchLoss:.3e}, Generator Loss: {generatorBatchLoss:.3e}, ' + \
                                        f'Discriminator Accuracy: {discriminatorBatchAccuracy:.2f}, Generator Accuracy: {generatorBatchAccuracy:.2f}')

        torch.save(self.discriminatorNet.state_dict(), weightsDirectoryPath + DiscriminatorPath + finalWeightsPath + weightsExtension)
        torch.save(self.generatorNet.state_dict(), weightsDirectoryPath + GeneratorPath + finalWeightsPath + weightsExtension)

        return discriminatorTrainingLoss, discriminatorAccuracy, generatorLoss, generatorAccuracy
