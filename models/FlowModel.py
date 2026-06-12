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

    def generate(self, x: torch.Tensor, rotation: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Forward pass through the flow matching model."""
        dt = 1.0 / num_steps
        curr = x
        for i in range(num_steps):
            t = torch.full((x.size(0), 1), i * dt, device=x.device)
            curr = curr + self.model(curr, rotation, t) * dt
        return curr

    def train_step(self, input: torch.Tensor, rotation: torch.Tensor, target: torch.Tensor, criterion: nn.Module):
        """Perform a single training step."""
        self.train()
        t = torch.rand(target.size(0), 1, device=target.device)

        x_t = (1 - t) * input + t * target
        v_target = target - input

        pred = self.model(x_t, rotation, t)
        loss = criterion(pred, v_target)
        return loss