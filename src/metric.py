from torchmetrics import Metric
# from torchmetrics.classification import MulticlassPrecisionRecallCurve
import torch

class MyF1Score(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # F1 계산을 위한 상태 초기화
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        
        # PR Curve 계산을 위한 내장 메트릭
        # self.pr_curve = MulticlassPrecisionRecallCurve(num_classes=num_classes)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """preds: [B, C] shape의 로짓 텐서, target: [B] shape의 정답 레이블"""
        # 클래스 예측값 계산
        pred_labels = torch.argmax(preds, dim=1)
        # assert preds.shape == target.shape
        
        # F1 통계량 누적
        for c in range(self.num_classes):
            self.tp[c] += ((pred_labels == c) & (target == c)).sum()
            self.fp[c] += ((pred_labels == c) & (target != c)).sum()
            self.fn[c] += ((pred_labels != c) & (target == c)).sum()
        
        # PR Curve 업데이트 (로짓 전달)
        # self.pr_curve.update(preds, target)

    def compute(self) -> tuple:
        """(macro_f1, pr_curve) 반환"""
        eps = 1e-9  # 0 나눗셈 방지
        
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # pr = self.pr_curve.compute()  # (precision, recall, thresholds)
        
        return f1.mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pred_labels = torch.argmax(preds, dim=1)
        assert pred_labels.shape == target.shape, "Unequal Pred and Target"
        
        self.correct += torch.sum(pred_labels == target)
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total.float()
