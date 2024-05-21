from network.ProGAN import ProGAN

def main():
    configuration = {"Discriminator": {"channels": [128, 256, 256, 512, 512, 512]},
                    "Generator": {"latentSpaceSize": 100, "channels": [512, 512, 512, 256, 256, 128]},
                    "Training": {"extraSteps": 1, "penaltyRatio": 10, "epsilonDrift": 1e-3,
                                 "epochUpscaling": [40, 80, 120, 160, 240], "epochAdaptation": 10,
                                 "batchSizes": {"128": 8, "64": 32, "32": 128, "16": 256, "8": 1024, "4": 2048}}}
    network = ProGAN(imageShape = (128, 128, 3), configuration = configuration)
    network.train_gan(numEpochs = 161, learningRate = 1e-4, continueTraining = True, currentResolution = 64, currentEpoch = 170)

if __name__ == "__main__":
    main()