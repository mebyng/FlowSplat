import torch
import torch.nn as nn

from .AutoEncoder import AutoEncoderLoss


class FlowWrapper(nn.Module):
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
            inp = torch.cat([curr, plucker], dim=1)
            curr = curr + self.model(inp, t) * dt
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
    


class RegressionWrapper(nn.Module):
    """Simple Regression model."""

    def __init__(self, model: nn.Module, mode: str = "rotate"):
        super().__init__()
        self.model = model
        self.mode = mode

        if self.mode in ["rotate", "generate"]:
            self.criterion = torch.nn.MSELoss()
        elif self.mode == "encode":
            self.criterion = AutoEncoderLoss(self.device())
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def device(self):
        """Return the device of the model parameters."""
        return self.model.device()
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
        

    def generate(self, input: torch.Tensor, plucker: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Forward pass through the regression model."""

        if self.mode == "rotate":
            input = torch.cat([input, plucker], dim=1)  # Concatenate input and target images for rotation mode
        elif self.mode == "generate":
            input = plucker  # Use Plucker coordinates directly for generation mode
        elif self.mode == "encode":
            input = input  # Use input image directly for encoding mode
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.mode == "encode":
            return self.model(input)[0]
        else:
            return self.model(input)

    def train_step(self, input: torch.Tensor, plucker: torch.Tensor, target: torch.Tensor):
        """Perform a single training step."""
        self.train()

        if self.mode == "rotate":
            input = torch.cat([input, plucker], dim=1)  # Concatenate input and target images for rotation mode
        elif self.mode == "generate":
            input = plucker  # Use Plucker coordinates directly for generation mode
        elif self.mode == "encode":
            input = input  # Use input image directly for encoding mode
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        pred = self.model(input)
        loss = self.criterion(pred, target)
        return loss