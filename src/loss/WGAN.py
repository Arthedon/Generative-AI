import torch
from torch import nn

class WassersteinDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(WassersteinDiscriminatorLoss, self).__init__()

    def forward(
            self, realDataDiscriminatorOutput: torch.Tensor, fakeDataDiscriminatorOutput: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the wasserstein loss between the fake data and real data.

        Parameters
        ----------
        fakeDataDiscriminatorOutput : torch.Tensor
            The output of the generator.
        realDataDiscriminatorOutput : torch.Tensor
            The image from the dataset.

        Returns
        -------
        torch.Tensor
            The computed wasserstein loss value.
        """
        return torch.mean(realDataDiscriminatorOutput) - torch.mean(fakeDataDiscriminatorOutput)
    
class WassersteinGeneratorLoss(nn.Module):
    def __init__(self):
        super(WassersteinGeneratorLoss, self).__init__()

    def forward(
            self, fakeDataDiscriminatorOutput: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the wasserstein loss between the fake data and real data.

        Parameters
        ----------
        fakeDataDiscriminatorOutput : torch.Tensor
            The output of the generator.
        realDataDiscriminatorOutput : torch.Tensor
            The image from the dataset.

        Returns
        -------
        torch.Tensor
            The computed wasserstein loss value.
        """
        return torch.mean(fakeDataDiscriminatorOutput)
    
class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(
            self, yTrue: torch.Tensor, yPred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the wasserstein loss between the fake data and real data.

        Parameters
        ----------
        yTrue : Real label (-1 for fake data and 1 for real data)
        yPred : Discriminator prediction

        Returns
        -------
        torch.Tensor
            The computed wasserstein loss value.
        """
        return torch.mean(yTrue * yPred)