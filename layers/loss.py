import torch
from torch import nn
import torch.nn.functional as F

class CustomCrossEntropy(nn.Module):
    def __init__(self):
        super(CustomCrossEntropy, self).__init__()

    def forward(self, y_pred, y_true):
        pad_event = -1
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], 3)
        y_true = y_true.view(-1)
        masks = (y_true != pad_event).float()
        y_true = y_true * masks.long()
        criterion = nn.NLLLoss(reduction="none")
        losses = criterion(torch.log(y_pred + 1e-12), y_true)
        loss = torch.mean(losses * masks)
        return loss


def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss