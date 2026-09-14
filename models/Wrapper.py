import torch
import torch.nn as nn

from losses import AutoEncoderLoss, MSELoss


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

    def generate(
        self,
        input: torch.Tensor,
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
        input: torch.Tensor,
        noise: torch.Tensor,
        cameras: torch.Tensor,
        num_steps: int = 10,
    ):
        """Log the generation process."""

        return self.generate(input, noise, cameras, num_steps), None

    def train_step(
        self, input: torch.Tensor, target: torch.Tensor, cameras: torch.Tensor
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
        input: torch.Tensor,
        noise: torch.Tensor,
        cameras: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Forward pass through the flow matching model."""
        dt = 1.0 / num_steps
        current = noise
        condition = input
        if hasattr(self.model, "encode_condition"):
            condition, _ = self.model.encode_condition(condition)
        for i in range(num_steps):
            t = torch.full((input.size(0), 1), i * dt, device=input.device)
            velocity = self.model(current, condition, cameras, t)
            current = current + velocity * dt
        return current

    def generation_log(
        self,
        input: torch.Tensor,
        noise: torch.Tensor,
        cameras: torch.Tensor,
        num_steps: int = 10,
    ):
        """Log the generation process."""

        dt = 1.0 / num_steps
        current = noise
        condition = input
        if hasattr(self.model, "encode_condition"):
            condition, _ = self.model.encode_condition(condition)

        intermediates = []

        for i in range(num_steps):
            t = torch.full((input.size(0), 1), i * dt, device=input.device)
            velocity = self.model(current, condition, cameras, t)
            intermediates.append(current + velocity * dt * (num_steps - i))
            current = current + velocity * dt

        return current, intermediates

    def train_step(
        self, input: torch.Tensor, target: torch.Tensor, cameras: torch.Tensor
    ):
        """Perform a single training step."""
        self.train()
        t = torch.rand(target.size(0), 1, device=target.device)
        noise = torch.randn_like(target)
        condition = input

        interp = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - interp) * noise + interp * target
        v_target = target - noise

        pred = self.model(x_t, condition, cameras, t)
        loss, loss_dict = self.criterion(pred, v_target)
        return loss, loss_dict


class RegressionModel(Wrapper):
    """Simple Regression model."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.criterion = MSELoss()

    def generate(
        self,
        input: torch.Tensor,
        noise: torch.Tensor,
        cameras: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Forward pass through the regression model."""

        # condition = torch.cat([input, rotation], dim=1)
        condition = input
        input = None

        return self.model(input, condition, cameras, None)

    def train_step(
        self, input: torch.Tensor, target: torch.Tensor, cameras: torch.Tensor
    ):
        """Perform a single training step."""
        self.train()

        # condition = torch.cat([input, rotation], dim=1)
        condition = input
        input = torch.zeros_like(input)

        pred = self.model(input, condition, cameras, None)
        loss, loss_dict = self.criterion(pred, target)
        return loss, loss_dict


class AutoEncoder(Wrapper):
    """Simple AutoEncoder model."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.criterion = AutoEncoderLoss(self.device())

    def generate(
        self,
        input: torch.Tensor,
        noise: torch.Tensor,
        cameras: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Forward pass through the autoencoder model."""

        return self.model(input)[0]

    def train_step(
        self, input: torch.Tensor, target: torch.Tensor, cameras: torch.Tensor
    ):
        """Perform a single training step."""
        self.train()
        pred = self.model(input)
        loss, loss_dict = self.criterion(pred, target)
        return loss, loss_dict
