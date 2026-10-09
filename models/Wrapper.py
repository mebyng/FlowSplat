import torch
import torch.nn as nn

from losses import AutoEncoderLoss, FlowDistillationLoss, MSELoss


class Wrapper(nn.Module):
    """Super Class for Wrapping Models with a common interface for training and generation."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def device(self):
        """Return the device of the model parameters."""
        return self.model.device()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def load_checkpoint(self, checkpoint, continued=False):
        """Restore model-owned state from an already loaded checkpoint."""
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return checkpoint

    def checkpoint_state(self):
        """Return model-owned state for a training checkpoint."""
        return {"model_state_dict": self.model.state_dict()}

    def generate(
        self,
        source: torch.Tensor,
        noise: torch.Tensor,
        cameras: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        raise NotImplementedError(
            "The generate method must be implemented in subclasses."
        )

    def generation_log(
        self,
        source: torch.Tensor,
        noise: torch.Tensor,
        cameras: torch.Tensor,
        num_steps: int = 10,
    ):
        """Log the generation process."""

        return self.generate(source, noise, cameras, num_steps), None

    def train_step(
        self, source: torch.Tensor, target: torch.Tensor, cameras: torch.Tensor
    ):
        """Perform a single training step."""
        raise NotImplementedError(
            "The train_step method must be implemented in subclasses."
        )


class FlowModel(Wrapper):
    """Simple Flow Matching model."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.criterion = MSELoss()

    def generate(
        self,
        source: torch.Tensor,
        noise: torch.Tensor,
        cameras: dict[str, torch.Tensor],
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Forward pass through the flow matching model."""
        dt = 1.0 / num_steps
        current = noise
        if hasattr(self.model, "encode_source"):
            source, _ = self.model.encode_source(source)
        for i in range(num_steps):
            t = torch.full((source.size(0), 1), i * dt, device=source.device)
            velocity = self.model(current, source, cameras, t)
            current = current + velocity * dt
        return current

    def generation_log(
        self,
        source: torch.Tensor,
        noise: torch.Tensor,
        cameras: dict[str, torch.Tensor],
        num_steps: int = 10,
    ):
        """Log the generation process."""

        dt = 1.0 / num_steps
        current = noise
        if hasattr(self.model, "encode_source"):
            source, _ = self.model.encode_source(source)

        intermediates = []

        for i in range(num_steps):
            t = torch.full((source.size(0), 1), i * dt, device=source.device)
            velocity = self.model(current, source, cameras, t)
            intermediates.append(current + velocity * dt * (num_steps - i))
            current = current + velocity * dt

        return current, intermediates

    def train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        cameras: dict[str, torch.Tensor],
    ):
        """Perform a single training step."""
        t = torch.rand(target.size(0), 1, device=target.device)
        noise = torch.randn_like(target)

        interp = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - interp) * noise + interp * target
        v_target = target - noise

        pred = self.model(x_t, source, cameras, t)
        loss, loss_dict = self.criterion(pred, v_target)
        return loss, loss_dict


class RegressionModel(Wrapper):
    """Simple Regression model."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.criterion = MSELoss()

    def generate(
        self,
        source: torch.Tensor,
        noise: torch.Tensor,
        cameras: dict[str, torch.Tensor],
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Forward pass through the regression model."""

        return self.model(None, source, cameras, None)

    def train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        cameras: dict[str, torch.Tensor],
    ):
        """Perform a single training step."""
        pred = self.model(torch.zeros_like(source), source, cameras, None)
        loss, loss_dict = self.criterion(pred, target)
        return loss, loss_dict


class AutoEncoder(Wrapper):
    """Simple AutoEncoder model."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.criterion = AutoEncoderLoss(self.device())

    def generate(
        self,
        source: torch.Tensor,
        noise: torch.Tensor,
        cameras: dict[str, torch.Tensor],
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Forward pass through the autoencoder model."""

        return self.model(source)[0]

    def train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        cameras: dict[str, torch.Tensor],
    ):
        """Perform a single training step."""
        pred = self.model(source)
        loss, loss_dict = self.criterion(pred, target)
        return loss, loss_dict


class GaussianModel(Wrapper):
    """Optimize one GaussianScene with a frozen pixel-space FlowModel teacher.

    Source image/pose and target cameras are supplied per step. No target
    images or teacher-generated views are used. Optimize self.model.parameters().
    """

    def __init__(self, model, teacher, min_t=0.05, max_t=0.95, depth_weight=0.1):
        super().__init__(model)
        self.teacher = teacher.to(model.device()).eval().requires_grad_(False).eval()
        self.min_t, self.max_t = min_t, max_t
        if teacher.model.in_channels not in (3, 4):
            raise ValueError("Gaussian supervision requires an RGB or RGB-D teacher")
        self.use_depth = teacher.model.in_channels == 4
        self.criterion = FlowDistillationLoss(depth_weight=depth_weight)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self, mode=True):
        self.model.eval()
        return self

    def render(self, source, cameras):
        return self.model.render(
            cameras["target"],
            cameras["intrinsics"],
            height=source.shape[-2],
            width=source.shape[-1],
        )

    def generate(self, source, noise, cameras, num_steps=10):
        return self.render(source, cameras)["rgb"]

    def train_step(self, source, target, cameras):
        """Distill RGB and, for RGB-D teachers, expected depth without alpha weighting."""
        expected_channels = 4 if self.use_depth else 3
        if source.shape[1] != expected_channels:
            raise ValueError(f"teacher requires {expected_channels}-channel source images")
        rendered = self.render(source, cameras)
        image = rendered["rgb"]
        if self.use_depth:
            image = torch.cat([image, rendered["depth"]], dim=1)
        batch_size = image.shape[0]
        noise = torch.randn_like(image)
        timestep = torch.rand(batch_size, 1, device=image.device, dtype=image.dtype)
        timestep = self.min_t + (self.max_t - self.min_t) * timestep
        with torch.no_grad():
            t = timestep[:, :, None, None]
            x_t = (1 - t) * noise + t * image.detach()
            velocity = self.teacher.model(
                x_t,
                source,
                cameras,
                timestep,
            )
        return self.criterion(image, noise, timestep, velocity)
