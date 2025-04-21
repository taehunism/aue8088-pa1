# Python packages
from termcolor import colored
from tqdm import tqdm
import os
import tarfile
import wget

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Custom packages
import src.config as cfg


class TinyImageNetDatasetModule(LightningDataModule):
    __DATASET_NAME__ = 'tiny-imagenet-200'

    def __init__(self, batch_size: int = cfg.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        if not os.path.exists(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__)):
            # download data
            print(colored("\nDownloading dataset...", color='green', attrs=('bold',)))
            filename = self.__DATASET_NAME__ + '.tar'
            wget.download(f'https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/{filename}')

            # extract data
            print(colored("\nExtract dataset...", color='green', attrs=('bold',)))
            with tarfile.open(name=filename) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(path=cfg.DATASET_ROOT_PATH, member=member)
            os.remove(filename)

    def train_dataloader(self):
        # tf_train = transforms.Compose([
        #     transforms.RandomRotation(cfg.IMAGE_ROTATION),
        #     transforms.RandomHorizontalFlip(cfg.IMAGE_FLIP_PROB),
        #     transforms.RandomCrop(cfg.IMAGE_NUM_CROPS, padding=cfg.IMAGE_PAD_CROPS),
        #     transforms.ToTensor(),
        #     transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        # ])
        
        tf_train = transforms.Compose([
            # 1. 기존 증강
            transforms.RandomRotation(cfg.IMAGE_ROTATION),
            transforms.RandomHorizontalFlip(cfg.IMAGE_FLIP_PROB),
            transforms.RandomResizedCrop(  # RandomCrop → RandomResizedCrop으로 변경
                size=64, 
                scale=(0.6, 1.0),  # 60%~100% 크기로 랜덤 리사이즈
                ratio=(0.8, 1.2)   # 가로세로 비율 범위
            ),
            
            # 2. 색상 증강 강화 (config.py의 COLOR_JITTER 사용)
            transforms.ColorJitter(**cfg.COLOR_JITTER),
            
            # 3. 텐서 변환
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
            
            # 4. RandomErasing 추가 (config.py의 RANDOM_ERASING 사용)
            transforms.RandomErasing(**cfg.RANDOM_ERASING)
        ])
    
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'train'), tf_train)
        msg = f"[Train]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'val'), tf_val)
        msg = f"[Val]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'test'), tf_test)
        msg = f"[Test]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )