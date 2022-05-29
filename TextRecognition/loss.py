# CTCLoss
import torch
import torch.nn as nn
import math

class ctcloss(nn.Module):
    def __init__(self):
        super(ctcloss, self).__init__()
        self.loss = nn.CTCLoss(zero_infinity=True)

    def forward(self, logits, target, pred_size, target_size):
        loss = self.loss(logits, target, pred_size, target_size)
        eps = 1e-7
        if abs(loss.item() - float('inf')) < eps:
            return torch.zeros_like(loss)
        if math.isnan(loss.item()):
            print("nan_loss")
            return torch.zeros_like(loss)
        return loss
    

