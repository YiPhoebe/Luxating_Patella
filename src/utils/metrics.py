import torch


@torch.no_grad()
def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def precision_recall_f1(outputs, targets, positive_class=1):
    preds = outputs.argmax(dim=1)
    tp = ((preds == positive_class) & (targets == positive_class)).sum().item()
    fp = ((preds == positive_class) & (targets != positive_class)).sum().item()
    fn = ((preds != positive_class) & (targets == positive_class)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1

