from torchvision.datasets import MNIST
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, data_dir, data_transform, labels=None):
        self.data_transform = data_transform

        # Just for illustration, I only load the training dataset and then split it.
        self.dataset = MNIST(root=data_dir, train=True, download=True)

        if labels is not None:
            # Filter dataset indices where label is in labels
            self.indices = [i for i, target in enumerate(self.dataset.targets) if target in labels]
        else:
            self.indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]
        image = self.data_transform(image)
        return image, label
