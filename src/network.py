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

import wandb

# [TODO: Optional] Rewrite this class if you want
# class MyNetwork(AlexNet):
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        # [TODO] Modify feature extractor part in AlexNet
       # 1. ResNet18 기본 구조 로드
        self.model = models.resnet18(weights=None)
        
        # 2. 64x64 입력을 위한 초기 레이어 수정
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # stride=2 복원
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # [TODO: Optional] Modify this as well if you want
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     return x

class SimpleClassifier(LightningModule):
    def __init__(self,
                #  model_name: str = 'resnet18',
                 model_name: str = 'MyResNet',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        # if model_name == 'MyNetwork':
        if model_name == 'MyResNet':
            self.model = CustomResNet18(num_classes).to(self.device)
        # if model_name == 'resnet18':
        #     self.model = models.resnet18(weights='IMAGENET1K_V1')
        #     self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy().to(self.device)
        self.train_f1 = MyF1Score(num_classes=num_classes).to(self.device)
        self.val_f1 = MyF1Score(num_classes=num_classes).to(self.device)
        
        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

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
        accuracy = self.accuracy(scores, y)
        # f1, pr = self.f1_score(scores, y)
        self.train_f1(scores, y)
        
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'f1/train' : self.train_f1},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
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