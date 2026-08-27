import torch
import torch.nn as nn

from losses import AutoEncoderLoss, MSELoss


class FlowModel(nn.Module):
    """Simple Flow Matching model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        self.criterion = MSELoss()

    def device(self):
        """Return the device of the model parameters."""
        return self.model.device()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def generate(
        self, input: torch.Tensor, rotation: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """Forward pass through the flow matching model."""
        dt = 1.0 / num_steps
        curr = input[:, 3:]
        cond = input[:, :3]
        for i in range(num_steps):
            t = torch.full((input.size(0), 1), i * dt, device=input.device)
            inp = torch.cat([curr, cond, rotation], dim=1)
            curr = curr + self.model(inp, t) * dt
        return curr

    def train_step(
        self, input: torch.Tensor, rotation: torch.Tensor, target: torch.Tensor
    ):
        """Perform a single training step."""
        self.train()
        t = torch.rand(target.size(0), 1, device=target.device)
        noise = input[:, 3:]
        cond = input[:, :3]

        interp = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - interp) * noise + interp * target
        v_target = target - noise

        x_t = torch.cat(
            [x_t, cond, rotation], dim=1
        )  # Concatenate input with rotation encoding

        pred = self.model(x_t, t)
        loss, loss_dict = self.criterion(pred, v_target)
        return loss, loss_dict


class RegressionModel(nn.Module):
    """Simple Regression model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        self.criterion = MSELoss()

    def device(self):
        """Return the device of the model parameters."""
        return self.model.device()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def generate(
        self, input: torch.Tensor, rotation: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """Forward pass through the regression model."""

        # condition = torch.cat([input, rotation], dim=1)
        condition = input
        input = torch.zeros_like(input)

        return self.model(input, condition, rotation[:, :, 0, 0])

    def train_step(
        self, input: torch.Tensor, rotation: torch.Tensor, target: torch.Tensor
    ):
        """Perform a single training step."""
        self.train()

        # condition = torch.cat([input, rotation], dim=1)
        condition = input
        input = torch.zeros_like(input)

        pred = self.model(input, condition, rotation[:, :, 0, 0])
        loss, loss_dict = self.criterion(pred, target)
        return loss, loss_dict


class AutoEncoder(nn.Module):
    """Simple AutoEncoder model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        self.criterion = AutoEncoderLoss(self.device())

    def device(self):
        """Return the device of the model parameters."""
        return self.model.device()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def generate(
        self, input: torch.Tensor, rotation: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """Forward pass through the autoencoder model."""

        return self.model(input)[0]

    def train_step(
        self, input: torch.Tensor, rotation: torch.Tensor, target: torch.Tensor
    ):
        """Perform a single training step."""
        self.train()
        pred = self.model(input)
        loss, loss_dict = self.criterion(pred, target)
        return loss, loss_dict
