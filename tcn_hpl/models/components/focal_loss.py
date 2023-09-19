import torch
from torch import nn

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss from the paper 
    `https://arxiv.org/pdf/1708.02002v2.pdf`
    """
    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        self.ce = nn.CrossEntropyLoss(
            ignore_index=-100, 
            reduction="none"
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        # Check reduction option and return loss accordingly
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()

        return focal_loss