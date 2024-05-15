from network.WGAN import WGAN

def main():
    configuration = {"Discriminator": {"channels": [16, 64, 128, 256, 512, 2048]},
                    "Generator": {"latentSpaceSize": 100, "channels": [4096, 1024, 512, 256, 128]},
                    "Training": {"extraSteps": 5, "penaltyRatio": 10, "minLearningRate": 1e-7}}
    network = WGAN(imageShape = (128, 128, 3), configuration = configuration)
    network.train_gan(numEpochs = 35, learningRate = 1e-4, batchSize = 128, continueTraining = True)

if __name__ == "__main__":
    main()