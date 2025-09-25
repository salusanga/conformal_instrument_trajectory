import torch
import torch.nn as nn

class AllQuantileLoss(nn.Module):
    """Pinball loss function with cosine similarity"""

    def __init__(self, quantiles):
        """Initialize

        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)


        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """Compute the pinball loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        errors = target.expand(-1, -1, preds.shape[-1]) - preds

        for q in self.quantiles:
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(2))
        return torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))


class CosineQuantileLoss(nn.Module):
    """Pinball loss function"""

    def __init__(self, quantiles):
        """Initialize

        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)


        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """Compute the pinball loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        angle_difference = torch.atan2(
            torch.sin(target - preds),
            torch.cos(target - preds),
        ).unsqueeze(-1)
        one_minus_cos_diff = 1 - torch.cos(angle_difference)
        for q in self.quantiles:
            losses.append(
                torch.max(
                    (q - 1) * one_minus_cos_diff, q * one_minus_cos_diff
                ).unsqueeze(2)
            )
        return torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))