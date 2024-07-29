import os
import random
import numpy as np
from PIL import Image
from robust_minisets.info import INFO, HOMEPAGE, DEFAULT_ROOT

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class RobustMiniset(Sequence):

    flag = ...

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        as_rgb=False,
        root=DEFAULT_ROOT,
        mmap_mode=None
    ):
        ''' 
        Args:

            split (str, required): 'train', 'val' or 'test'
            transform (callable, optional): data transformation
            target_transform (callable, optional): target transformation
            mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.

        '''

        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(
            os.path.join(self.root, f"{self.flag}.npz"),
            mmap_mode=mmap_mode,
        )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split in ["train", "val", "test"]:
            self.imgs = npz_file[f"{self.split}_images"]
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError

    def __len__(self):
        return self.imgs.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.ss'''
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Resolution: {self.info['resolution']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info["url"],
                root=self.root,
                filename=f"{self.flag}.npz",
                md5=self.info["MD5"],
            )
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               HOMEPAGE)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(np.array(x))
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="png", write_csv=True):

        from robust_minisets.utils import save2d

        save2d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}.csv") if write_csv else None
        )

    def montage(self, length=20, replace=False, save_folder=None):
        from robust_minisets.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(imgs=self.imgs,
                                n_channels=self.info['n_channels'],
                                sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(os.path.join(save_folder,
                                          f"{self.flag}_{self.split}_montage.jpg"))

        return montage_img


# /////////////// CIFAR ///////////////

class CIFAR10_1(RobustMiniset):
    flag = "cifar-10-1"

class CIFAR10C(RobustMiniset):
    flag = "cifar-10-c"

class CIFAR100C(RobustMiniset):
    flag = "cifar-100-c"

# /////////////// EuroSAT ///////////////

class EuroSAT(RobustMiniset):
    flag = "eurosat"

class EuroSATC(RobustMiniset):
    flag = "eurosat-c"

# /////////////// Tiny ImageNet ///////////////

class TinyImageNet(RobustMiniset):
    flag = "tiny-imagenet"

class TinyImageNetA(RobustMiniset):
    flag = "tiny-imagenet-a"

class TinyImageNetC(RobustMiniset):
    flag = "tiny-imagenet-c"

class TinyImageNetR(RobustMiniset):
    flag = "tiny-imagenet-r"

class TinyImageNetv2(RobustMiniset):
    flag = "tiny-imagenet-v2"
        
# /////////////// MedMNIST ///////////////

class BloodMNISTC(RobustMiniset):
    flag = "bloodmnist-c"

class BreastMNISTC(RobustMiniset):
    flag = "breastmnist-c"

class DermaMNISTC(RobustMiniset):
    flag = "dermamnist-c"

class OCTMNISTC(RobustMiniset):
    flag = "octmnist-c"

class OrganAMNISTC(RobustMiniset):
    flag = "organamnist-c"

class OrganCMNISTC(RobustMiniset):
    flag = "organcmnist-c"

class OrganSMNISTC(RobustMiniset):
    flag = "organsmnist-c"

class PathMNISTC(RobustMiniset):
    flag = "pathmnist-c"

class PneumoniaMNISTC(RobustMiniset):
    flag = "pneumoniamnist-c"

class TissueMNISTC(RobustMiniset):
    flag = "tissuemnist-c"


def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)
