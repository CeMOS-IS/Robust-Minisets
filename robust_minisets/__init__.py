from robust_minisets.info import __version__, HOMEPAGE, INFO
try:
    from robust_minisets.dataset import (CIFAR10_1, CIFAR10C, CIFAR100C, EuroSAT, EuroSATC, BloodMNISTC, BreastMNISTC,
                                         DermaMNISTC, OCTMNISTC, OrganAMNISTC, OrganCMNISTC, OrganSMNISTC, PathMNISTC,
                                         PneumoniaMNISTC, TissueMNISTC, TinyImageNet, TinyImageNetA, TinyImageNetC,
                                         TinyImageNetR, TinyImageNetv2)
except:
    print("Please install the required packages first. " +
          "Use `pip install -r requirements.txt`.")
