from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('true_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        
        # 벡터화 연산으로 성능 개선
        for c in range(self.num_classes):
            self.true_positives[c] += ((preds == c) & (target == c)).sum()
            self.false_positives[c] += ((preds == c) & (target != c)).sum()
            self.false_negatives[c] += ((preds != c) & (target == c)).sum()

    def compute(self):
        eps = 1e-9
        precision = self.true_positives / (self.true_positives + self.false_positives + eps)
        recall = self.true_positives / (self.true_positives + self.false_negatives + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        return f1.mean()  # Macro F1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")   

        # [TODO] Count the number of correct prediction
        correct = torch.sum(preds == target)
        
        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
