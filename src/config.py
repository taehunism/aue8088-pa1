import os

# Training Hyperparameters
NUM_CLASSES         = 200 # 분류할 클래스의 개수
BATCH_SIZE          = 512 # 모델에 입력되는 샘플 수 , 크면 학습이 안정적인데 , GPU 사용량 증가
VAL_EVERY_N_EPOCH   = 1 # n 에폭 마다 validation 진행

NUM_EPOCHS          = 40 # 총 에폭

OPTIMIZER_PARAMS = {
    'type': 'Adam',
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-3
}

SCHEDULER_PARAMS = {
    'type': 'MultiStepLR',
    'milestones': [25, 35],
    'gamma': 0.1
}

# OPTIMIZER_PARAMS    = {
#     'type': 'AdamW',
#     'lr': 3e-4,
#     'weight_decay': 1e-4,  # L2 정규화 추가
# }

# SCHEDULER_PARAMS    = {
#     'type': 'CosineAnnealingLR',
#     'T_max': 40,
#     'eta_min': 1e-6
# }

# SCHEDULER_PARAMS = {
#     'type': 'OneCycleLR',  # Cosine → OneCycleLR로 변경
#     'max_lr': 4e-4,
#     'total_steps': NUM_EPOCHS * (100000//BATCH_SIZE),
#     'pct_start': 0.3,
# }

# OPTIMIZER_PARAMS    = {
#     'type': 'SGD',
#     'lr': 0.1,
#     'momentum':0.9,
#     'weight_decay': 5e-4,
#     'nesterov':True    
# }

#마일스톤 50 에폭과 75 에폭에 러닝 레이트 gamma배로 줄임 -> 총 에폭보다 낮은 마일 스톤 써야함

# OPTIMIZER_PARAMS = {
#     'type': 'SGD',
#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
# }

# Model parameters
DROPOUT_RATE       = 0.3
LABEL_SMOOTHING    = 0.1

# Data Augmentation parameters
COLOR_JITTER       = {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1
}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8 #데이터 로딩 할 때 사용할 subprocess 수 , 시스템 리소스 고려

# Augmentation
IMAGE_ROTATION      = 30 # 이미지 랜덤 회전 각도 범위
IMAGE_FLIP_PROB     = 0.7 # 좌우 반전 확률
IMAGE_NUM_CROPS     = 64 #64 # Crop 생성할 횟수
IMAGE_PAD_CROPS     = 8 #4 패딩 후 Crop을 수행 할 때 padding의 크기
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975] # 이미지 정규화에 사용하는 mean/std
IMAGE_STD           = [0.2302, 0.2265, 0.2262] # Tiny ImageNet의 RGB 채널 별 평균과 표준편차

RANDOM_ERASING = {
    'p': 0.5,
    'scale': (0.02, 0.33),
    'ratio': (0.3, 3.3)
}

# Network
# MODEL_NAME          = 'resnet18' #torchvision의 resnet18 사용
#MODEL_NAME          = 'mynet' #my network
MODEL_NAME = 'MyTinyNet'

# Compute related
ACCELERATOR         = 'gpu' # CUDA 가속기 
DEVICES             = 1 # 0번 gpu, 여러개면 [0,1,n-1,n]
PRECISION_STR       = '32-true' # 학습 정밀도 
# PRECISION_STR       = '16-mixed' # 학습 정밀도 

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = 'taehontas'
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
