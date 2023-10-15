from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class DLoader:
    def __init__(self, dataset: str = 'MNIST', bs: int = 64):
        self.train, self.val = self.get_dataset(dataset, bs)

    @staticmethod
    def get_dataset(dataset, bs):
        match dataset:
            case 'MNIST':
                from torchvision.datasets import MNIST
                train = MNIST(f'/data/{dataset}', train=True, download=True, transform=ToTensor())
                val = MNIST(f'/data/{dataset}', train=False, download=True, transform=ToTensor())
                return DataLoader(train, bs, shuffle=True), DataLoader(val, bs, shuffle=True)
