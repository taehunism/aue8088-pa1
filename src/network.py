# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger

from torch import nn
import torch.nn as nn

from torchvision import models
from torchvision.models.alexnet import AlexNet
from torchvision.models import resnet18
import torch

# Custom packages
from src.metric import MyAccuracy, MyF1Score 
import src.config as cfg
from src.util import show_setting
from ptflops import get_model_complexity_info
import wandb

# [TODO: Optional] Rewrite this class if you want
class MyTinyNet(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 64 → 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 → 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 16 → 8

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 8 → 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 4 → 2

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),# 2 → 1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 마지막 MaxPool 생략
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 1024),  # Feature map size가 1x1임!
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.model = models.resnet18(weights=None)

        # 입력 해상도 64x64에 적합하게 conv1 stride 수정
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 64 → 32
        self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                # 32 → 16

        # 분류기 수정
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
def compute_flops(model: nn.Module, input_res=(3, 64, 64)):
    model.eval()
    with torch.cuda.device(0):  # CUDA 디바이스에서 계산 (없으면 삭제)
        macs, params = get_model_complexity_info(
            model,
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
    flops = macs * 2  # 1 MAC = 2 FLOPs
    return flops, params

class SimpleClassifier(LightningModule):
    def __init__(self,
                # model_name: str = 'resnet18',
                #  model_name: str = 'MyTinyNet',
                model_name: str = cfg.MODEL_NAME,
                num_classes: int = 200,
                optimizer_params: Dict = dict(),
                scheduler_params: Dict = dict(),
        ):
        super().__init__()
        # Network
        if model_name == 'MyTinyNet':
            self.model = MyTinyNet(num_classes).to(self.device)
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
        elif model_name.startswith('efficientnet'):
            self.model = getattr(models, model_name)(weights=None)

            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.train_accuracy = MyAccuracy().to(self.device)
        self.val_accuracy = MyAccuracy().to(self.device)
        
        self.train_f1 = MyF1Score(num_classes=num_classes).to(self.device)
        self.val_f1 = MyF1Score(num_classes=num_classes).to(self.device)
        
        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)
        
        if isinstance(self.logger, WandbLogger):
            try:
                dummy_input_shape = (3, 64, 64)
                flops, params = compute_flops(self.model, dummy_input_shape)

                self.logger.experiment.summary["FLOPs"] = flops
                self.logger.experiment.summary["Params"] = params
                self.print(colored(f"[FLOPs] {flops/1e6:.2f} MFLOPs | [Params] {params/1e6:.2f} M", "green"))
                
            except Exception as e:
                self.print(colored(f"[FLOPs 계산 실패] {e}", "red"))

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.train_accuracy(scores, y)
        # f1, pr = self.f1_score(scores, y)
        self.train_f1(scores, y)
        
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'f1/train' : self.train_f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.val_accuracy(scores, y)
        # f1, pr = self.f1_score(scores, y)
        self.val_f1(scores, y)
        
        self.log_dict({
            'loss/val': loss,
            'accuracy/val': accuracy,
            'f1/val': self.val_f1
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # if isinstance(self.logger, WandbLogger):
        # # 1. GPU 상에서 확률값 계산 (메모리 효율적)
        #     y_proba = torch.softmax(scores, dim=1)  # [B, C] on GPU
            
        #     # 2. 첫 배치에서만 샘플링 (메모리 절약)
        #     if batch_idx == 0:
        #         # GPU → CPU 이동 (필요한 부분만 선택적 이동)
        #         sample_proba = y_proba[:5].detach().cpu().numpy()  # 상위 5개 샘플
        #         sample_true = y[:5].detach().cpu().numpy()
                
        #         # 3. W&B 호환 포맷 (전체 클래스 대신 Top-5 클래스만 시각화)
        #         top_classes = sample_proba.argmax(axis=1)
        #         class_labels = [f"class_{c}" for c in top_classes]
                
        #         self.logger.experiment.log({
        #             "pr_curve": wandb.plot.pr_curve(
        #                 y_true=sample_true,
        #                 y_probas=sample_proba,
        #                 labels=class_labels  # 실제 클래스명으로 교체 권장
        #             )
        #         })
        
        self._wandb_log_image(batch, batch_idx, scores, frequency=cfg.WANDB_IMG_LOG_FREQ)
            
    def _common_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0]],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
            
    def on_train_epoch_end(self):
        self.train_accuracy.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()
        self.val_f1.reset()
        
        if isinstance(self.logger, WandbLogger):
            # 현재 epoch의 평균 accuracy
            current_acc = self.trainer.callback_metrics.get("accuracy/val")
            prev_best = self.logger.experiment.summary.get("best_val_acc", 0.0)
            self.logger.experiment.summary["best_val_f1"] = self.trainer.callback_metrics.get("f1/val")
            self.logger.experiment.summary["best_val_loss"] = self.trainer.callback_metrics.get("loss/val")

            # current_acc가 더 높으면 best_val_acc 업데이트
            if current_acc is not None and current_acc > prev_best:
                self.logger.experiment.summary["best_val_acc"] = current_acc
            