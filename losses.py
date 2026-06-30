import torch
import torch.nn as nn
import lpips


class MSELoss(nn.Module):
    """Simple MSE loss that returns a loss dictionary for logging."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        loss = self.mse(prediction, target).mean()
        loss_dict = {"mse": loss, "total": loss}
        return loss, loss_dict


class AutoEncoderLoss(nn.Module):
    """Loss function for autoencoder training, combining MSE, LPIPS, and latent regularization."""

    def __init__(
        self,
        device,
        mse_weight: float = 1.0,
        lpips_weight: float = 0.0,
        latent_weight: float = 1e-3,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.latent_weight = latent_weight
        self.mse = nn.MSELoss()
        if lpips_weight != 0.0:
            self.lpips = lpips.LPIPS(net="vgg").to(device)

    def _latent_regularization(self, latent: torch.Tensor) -> torch.Tensor:
        mean = latent.mean(dim=[0, 2, 3])
        var = latent.var(dim=[0, 2, 3], unbiased=False)
        mean_loss = mean.pow(2).mean()
        var_loss = (var - 1.0).pow(2).mean()
        return mean_loss + var_loss

    def forward(self, encoding, target):
        reconstruction, latent = encoding

        loss_dict = {}

        loss_dict["mse"] = self.mse(reconstruction, target).mean()
        loss = self.mse_weight * loss_dict["mse"]

        if self.lpips_weight != 0.0:
            loss_dict["lpips"] = self.lpips(
                reconstruction, target, normalize=True
            ).mean()  # TODO change this when we add normalization to dataset
            loss = loss + self.lpips_weight * loss_dict["lpips"]

        if self.latent_weight != 0.0:
            loss_dict["latent_regulatization"] = self._latent_regularization(
                latent
            ).mean()
            loss = loss + self.latent_weight * loss_dict["latent_regulatization"]

        loss_dict["total"] = loss

        return loss, loss_dict
