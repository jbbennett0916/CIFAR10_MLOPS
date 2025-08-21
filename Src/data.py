from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# these are the average pixel intensities for the R,G,B channels of the CIFAR-10 dataset
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# these are the standard deviations of pixel intensities for the R,G,B channels of the CIFAR-10 dataset
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(train=True):
    """
    Returns a torchvision transform pipeline for CIFAR-10 preprocessing.

    Args:
        train (bool, optional): If True, returns transforms for training data
            including data augmentation. If False, returns transforms for test data.
            Defaults to True.

    Returns:
        torchvision.transforms.Compose: Composed transform pipeline for preprocessing images.
    """
    #
    if train: 
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.randomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])


def get_dataloader(root="data", batch_size=32, num_workers=4):
    train_dataset = datasets.CIFAR10(root, download=True, train=True, transforms=get_transforms())
    test_dataset = datasets.CIFAR10(root, download=True, train=False, transforms=get_transforms(train=False))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    return train_dataloader, test_dataloader, train_dataset.classes


if __name__ == "__main__":
    pass