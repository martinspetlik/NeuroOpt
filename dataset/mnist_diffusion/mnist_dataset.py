from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torch.utils.data import Subset


class MNISTDataset(Dataset):
    def __init__(self, data_dir, data_transform):
        self.data_transform = data_transform

        # Just for illustration, I only load the training dataset and then split it.
        self.dataset = MNIST(root=data_dir, train=True, download=True)

        print("self.dataset ", self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.data_transform(image)
        return image, label
