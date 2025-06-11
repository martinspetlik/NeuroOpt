import os
import joblib
import torchvision.transforms as transforms
from dataset.mnist_diffusion.mnist_dataset import MNISTDataset
from torch.utils.data import random_split
from model.auxiliary_functions import ScaleToMinusOneToOne, InverseScaleToZeroOne


def prepare_dataset(study, config, data_dir, serialize_path=None):
    ###
    # Data transforms
    ###
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        ScaleToMinusOneToOne()  # Scales to [-1, 1]
    ])

    # Inverse transform
    inverse_transform = InverseScaleToZeroOne()

    dataset = MNISTDataset(data_dir=data_dir, data_transform=transform)

    ####
    # Split datasets
    ####
    n_val_samples = int(config["n_train_samples"] * config["val_samples_ratio"])
    rest = len(dataset) - config["n_train_samples"] - n_val_samples - config["n_test_samples"]  # to avoid errors
    train_dataset, val_dataset, test_dataset, rest = random_split(dataset,
                                                                  [config["n_train_samples"], n_val_samples,
                                                                   config["n_test_samples"], rest])

    # Save attrs necessary for postprocessing
    study.set_user_attr("data_dir", data_dir)

    # It might be useful store datasets for further use
    if serialize_path is not None:
        joblib.dump(transform, os.path.join(serialize_path, "transform.pkl"))
        joblib.dump(inverse_transform, os.path.join(serialize_path, "inverse_transform.pkl"))
        joblib.dump(train_dataset, os.path.join(serialize_path, "train_dataset.pkl"))
        joblib.dump(val_dataset, os.path.join(serialize_path, "val_dataset.pkl"))
        joblib.dump(test_dataset, os.path.join(serialize_path, "test_dataset.pkl"))

    return train_dataset, val_dataset, test_dataset
