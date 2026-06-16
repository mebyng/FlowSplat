import torch
import torch.nn as nn


class FlowMatching(nn.Module):
    """Simple Flow Matching model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def device(self):
        """Return the device of the model parameters."""
        return self.model.device()

    def generate(self, input: torch.Tensor, plucker: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Forward pass through the flow matching model."""
        dt = 1.0 / num_steps
        curr = input
        for i in range(num_steps):
            t = torch.full((input.size(0), 1), i * dt, device=input.device)
            curr = curr + self.model(curr, plucker, t) * dt
        return curr

    def train_step(self, input: torch.Tensor, plucker: torch.Tensor, target: torch.Tensor, criterion: nn.Module):
        """Perform a single training step."""
        self.train()
        t = torch.rand(target.size(0), 1, device=target.device)

        interp = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - interp) * input + interp * target
        v_target = target - input

        x_t = torch.cat([x_t, plucker], dim=1)  # Concatenate input with Plucker coordinates

        pred = self.model(x_t, t)
        loss = criterion(pred, v_target)
        return loss
    


class DebugRegression(nn.Module):
    """Simple Regression model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def device(self):
        """Return the device of the model parameters."""
        return self.model.device()

    def generate(self, input: torch.Tensor, plucker: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Forward pass through the regression model."""
        return self.model(plucker)

    def train_step(self, input: torch.Tensor, plucker: torch.Tensor, target: torch.Tensor, criterion: nn.Module):
        """Perform a single training step."""
        self.train()

        pred = self.model(plucker)
        loss = criterion(pred, target)
        return loss