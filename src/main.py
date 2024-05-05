from network.GAN import GAN

def main():
    architecture = {"Discriminator": {"numClasses": 1, "channels": [16, 64, 256, 1024, 2048]}, "Generator": {"latentSpaceSize": 100, "channels":[4096, 1024, 512, 256, 128]}}
    network = GAN(imageShape = (128, 128, 3), architecture = architecture)
    network.train_gan(numEpochs = 75, learningRate = 2e-5, batchSize = 128, continueTraining = False)

if __name__ == "__main__":
    main()