import torch
import torch.nn as nn
import lpips


class MSELoss(nn.Module):
    """Simple MSE loss that returns a loss dictionary for logging."""

    def __init__(self, color_weight=1.0, depth_weight=0.1, use_depth=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.color_weight = color_weight
        self.depth_weight = depth_weight
        self.use_depth = use_depth

    def forward(self, prediction, target):
        if self.use_depth:
            pred_color = prediction[:, :3]
            target_color = target[:, :3]
        else:
            pred_color = prediction
            target_color = target
        color_loss = self.mse(pred_color, target_color).mean()
        loss = color_loss * self.color_weight
        loss_dict = {"mse": color_loss}
        if self.use_depth:
            pred_depth = prediction[:, 3:4]
            target_depth = target[:, 3:4]
            depth_loss = self.mse(pred_depth, target_depth).mean()
            loss = loss + depth_loss * self.depth_weight
            loss_dict["depth"] = depth_loss
        loss_dict["total"] = loss
        return loss, loss_dict


class AutoEncoderLoss(nn.Module):
    """Loss function for autoencoder training, combining MSE, LPIPS, and latent regularization."""

    def __init__(
        self,
        device,
        color_weight: float = 1.0,
        depth_weight: float = 0.1,
        lpips_weight: float = 0.0,
        latent_weight: float = 1e-3,
    ):
        super().__init__()
        self.color_weight = color_weight
        self.depth_weight = depth_weight
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

        pred_color = reconstruction[:, :3]
        target_color = target[:, :3]
        color_loss = self.mse(pred_color, target_color).mean()
        loss = color_loss * self.color_weight
        if reconstruction.shape[1] == 4:
            pred_depth = reconstruction[:, 3:4]
            target_depth = target[:, 3:4]
            depth_loss = self.mse(pred_depth, target_depth).mean()
            loss = loss + depth_loss * self.depth_weight
            loss_dict["depth"] = depth_loss

        loss_dict["mse"] = color_loss

        if self.lpips_weight != 0.0:
            loss_dict["lpips"] = self.lpips(pred_color, target_color).mean()
            loss = loss + self.lpips_weight * loss_dict["lpips"]

        if self.latent_weight != 0.0:
            loss_dict["latent_regularization"] = self._latent_regularization(
                latent
            ).mean()
            loss = loss + self.latent_weight * loss_dict["latent_regularization"]

        loss_dict["total"] = loss

        return loss, loss_dict


class FlowDistillationLoss(nn.Module):
    """SDS surrogate for the noise-to-image linear flow used by FlowModel.

    For x_t=(1-t)*noise+t*image and v= image-noise, the implied noise
    prediction is x_t-t*v. Thus eps_pred-noise = t*(image-noise-v).
    Use unit SDS weighting and detach the teacher/residual, bypassing its
    Jacobian. The scalar is a gradient surrogate, not reconstruction quality.
    """

    def __init__(self, depth_weight=0.1, use_encoding=False):
        super().__init__()
        if depth_weight < 0:
            raise ValueError("depth_weight must be nonnegative")
        self.depth_weight = depth_weight
        self.use_encoding = use_encoding

    def forward(self, image, noise, timestep, velocity):
        with torch.no_grad():
            gradient = timestep[:, :, None, None] * (image - noise - velocity)
            if not torch.isfinite(gradient).all():
                raise ValueError("nonfinite distillation gradient")
            target = image - gradient
        residual = image - target
        if self.use_encoding:
            loss = 0.5 * residual.square().mean()
            return loss, {
                "distillation_latent": loss.detach(),
                "distillation_gradient_rms": gradient.square().mean().sqrt(),
                "timestep": timestep.detach().mean(),
                "total": loss.detach(),
            }
        rgb_loss = 0.5 * residual[:, :3].square().mean()
        loss = rgb_loss
        metrics = {
            "distillation_rgb": rgb_loss.detach(),
            "distillation_gradient_rms": gradient.square().mean().sqrt(),
            "timestep": timestep.detach().mean(),
        }
        if image.shape[1] == 4:
            depth_loss = 0.5 * residual[:, 3:4].square().mean()
            loss = loss + self.depth_weight * depth_loss
            metrics["distillation_depth"] = depth_loss.detach()
        metrics.update(total=loss.detach())
        return loss, metrics
