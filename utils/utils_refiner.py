import torch
import torch.nn.functional as F

class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def custommse_loss(self, input: torch.Tensor, target: torch.Tensor):
        loss = F.mse_loss(input, target, reduction="none")
        # avg across space
        reduced_loss = torch.mean(loss, dim=tuple(range(3, loss.ndim)))
        # sum across time + fields
        reduced_loss = reduced_loss.sum(dim=(1, 2))
        # reduce along batch
        if self.reduction == "mean":
            return torch.mean(reduced_loss)
        elif self.reduction == "sum":
            return torch.sum(reduced_loss)
        elif self.reduction == "none":
            return reduced_loss
        else:
            raise NotImplementedError(self.reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.custommse_loss(input, target)
    
class ExponentialMovingAverage:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, overwrite=False):
        if len(self.shadow) > 0 and not overwrite:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.detach() + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        if len(self.shadow) == 0:
            print("Warning: EMA shadow is empty. Cannot apply shadow.")
        else:
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.data
                    param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def REL_L2(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Relative L2 (%) between predicted and true tensors."""
    if y.ndim == 3:
        y = y.unsqueeze(1)
    if y_hat.ndim == 3:
        y_hat = y_hat.unsqueeze(1)
    return (torch.mean((y - y_hat) ** 2) / torch.mean(y ** 2)).sqrt() * 100.0

def MSE(Yp: torch.Tensor, Yt: torch.Tensor):
    """Compute Mean Squared Error (MSE) between predicted and true tensors."""
    if Yt.ndim == 3:
        Yt = Yt.unsqueeze(1)
    if Yp.ndim == 3:
        Yp = Yp.unsqueeze(1)

    # Mean Squared Error
    return torch.mean((Yt - Yp) ** 2).item()  # true average over all dims

def REL_L1(Yp: torch.Tensor, Yt: torch.Tensor) -> torch.Tensor:
    """Compute Relative L1 (%) between predicted and true tensors."""
    if Yt.ndim == 3:
        Yt = Yt.unsqueeze(1)
    if Yp.ndim == 3:
        Yp = Yp.unsqueeze(1)
    return (torch.mean(torch.abs(Yt - Yp)) / torch.mean(torch.abs(Yt))) * 100.0