# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from thop import profile

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
class MyTinyNet(nn.Module):
    def __init__(self, num_classes=200, dropout_rate=0.3):
        super().__init__()

        class SEBlock(nn.Module):
            def __init__(self, channels, reduction=8):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(channels, channels // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels // reduction, channels),
                    nn.Sigmoid()
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                w = self.fc(self.pool(x)).view(b, c, 1, 1)
                return x * w
        
        class LightSqueezeExcitation(nn.Module):
            def __init__(self, channels, reduction=8):
                super().__init__()
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(channels, channels // reduction),
                    nn.Linear(channels // reduction, channels),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                b, c, _, _ = x.size()
                w = self.fc(self.global_pool(x)).view(b, c, 1, 1)
                return x * w

        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=7):
                super().__init__()
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
                
            def forward(self, x):
                # 채널 차원에서 평균 및 최대값 추출
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                # 두 특징 맵 연결(2채널 생성)
                x_cat = torch.cat([avg_out, max_out], dim=1)
                # 컨볼루션 + 시그모이드로 공간적 어텐션 맵 생성
                attention = torch.sigmoid(self.conv(x_cat))
                # 원본 입력과 곱하여 중요 영역 강조
                return x * attention

        class ChannelShuffle(nn.Module):
            def __init__(self, groups=4):
                super().__init__()
                self.groups = groups

            def forward(self, x):
                b, c, h, w = x.size()
                x = x.view(b, self.groups, c // self.groups, h, w)
                x = x.transpose(1, 2).contiguous()
                return x.view(b, c, h, w)

        class StochasticDepth(nn.Module):
            def __init__(self, p=0.2):
                super().__init__()
                self.p = p

            def forward(self, x):
                if not self.training or torch.rand(1)[0] > self.p:
                    return x
                return x * 0.5

        def depthwise_conv(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True)  # Swish-variant
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            LightSqueezeExcitation(64),  # SE 대신 LE 적용
            SpatialAttention(),  # 공간적 어텐션 추가
            
            depthwise_conv(64, 128, stride=2),  # 64x64 → 32x32
            LightSqueezeExcitation(128),  # SE 대신 LE 적용
            SpatialAttention(),  # 공간적 어텐션 추가
            
            depthwise_conv(128, 256, stride=2),  # 32x32 → 16x16
            LightSqueezeExcitation(256),  # SE 대신 LE 적용
            SpatialAttention(),  # 공간적 어텐션 추가
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            LightSqueezeExcitation(512),  # 어텐션 추가
            SpatialAttention(),  # 공간적 어텐션 추가
            ChannelShuffle(groups=4),
            StochasticDepth(p=0.3),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class LightSqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.fc(self.global_pool(x).view(b, c))
        return x * weights.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(x_att))

class ChannelShuffle(nn.Module):
    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, self.groups, c//self.groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, -1, h, w)

class StochasticDepth(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or torch.rand(1)[0] > self.p:
            return x
        return x * 0.5

class EnhancedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            LightSqueezeExcitation(planes)  # LE 통합
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            ChannelShuffle(groups=4)  # 채널 셔플
        )
        
        self.spatial_att = SpatialAttention()
        self.stochastic_depth = StochasticDepth(p=0.2)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.stochastic_depth(out) + identity
        out = self.spatial_att(out)  # 공간 어텐션 적용
        return self.relu(out)

def ResNet18_Modified(num_classes=200):
    class CustomResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                LightSqueezeExcitation(64)  # 초기 레이어에도 LE 적용
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # 레이어 스택 구성
            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * EnhancedBasicBlock.expansion, num_classes)

        def _make_layer(self, inplanes, planes, blocks, stride):
            downsample = None
            if stride != 1 or self.inplanes != planes * EnhancedBasicBlock.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * EnhancedBasicBlock.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * EnhancedBasicBlock.expansion),
                )

            layers = []
            layers.append(EnhancedBasicBlock(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * EnhancedBasicBlock.expansion
            for _ in range(1, blocks):
                layers.append(EnhancedBasicBlock(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return CustomResNet()
    
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
            # EfficientNet의 분류기 레이어 교체
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

        # FLOPs 계산 및 wandb에 기록
        # try:
        #     dummy_input = torch.randn(1, 3, 64, 64).to(next(self.model.parameters()).device)
        #     flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
        #     if isinstance(self.logger, WandbLogger):
        #         self.logger.experiment.summary['flops'] = flops
        #         self.logger.experiment.summary['params'] = params
        #         self.logger.experiment.config.update({'model_name': cfg.MODEL_NAME})
        #     print(f"[INFO] Model FLOPs: {flops:,}, Params: {params:,}")
        # except Exception as e:
        #     print(f"[FLOPs 계산 오류] {e}")
        try:
            dummy_input = torch.randn(1, 3, 64, 64).to(next(self.model.parameters()).device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            flops_m = flops / 1e6  # FLOPS in Millions
            params_m = params / 1e6  # Params in Millions

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.summary['FLOPS [M]'] = round(flops_m, 2)
                self.logger.experiment.summary['Params [M]'] = round(params_m, 2)
                self.logger.experiment.config.update({'model_name': cfg.MODEL_NAME})

            print(f"[INFO] Model FLOPS: {flops_m:.2f}M, Params: {params_m:.2f}M")
        except Exception as e:
            print(f"[FLOPS 계산 오류] {e}")


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