import os
import shutil
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

class TinyImageNetDataset(ImageFolder):
    """TinyImageNet-200 공식 데이터셋 클래스"""
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'

    def __init__(self, root, split='train', download=True, **kwargs):
        self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ('train', 'val'))
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download')
        
        super().__init__(self.split_dir, **kwargs)

    @property
    def dataset_dir(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def split_dir(self):
        return os.path.join(self.dataset_dir, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_dir)

    def download(self):
        if self._check_exists():
            return
        
        download_and_extract_archive(
            self.url, self.root, filename=self.filename,
            remove_finished=True, md5=self.zip_md5
        )
        
        # 검증셋 구조 정규화
        if self.split == 'val':
            self._normalize_val_structure()

    def _normalize_val_structure(self):
        val_dir = os.path.join(self.dataset_dir, 'val')
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        
        with open(annotations_file) as f:
            for line in f:
                img_name, label, *_ = line.strip().split('\t')
                img_src = os.path.join(val_dir, 'images', img_name)
                label_dir = os.path.join(val_dir, label)
                
                os.makedirs(label_dir, exist_ok=True)
                shutil.move(img_src, os.path.join(label_dir, img_name))
