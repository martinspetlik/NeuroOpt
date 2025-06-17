import os
import sys
import argparse
import joblib
import torch
import numpy as np
from visualization.visualize_data import plot_train_valid_loss, plot_sample
from model.mnist_diffusion.UNet import MnistUNet
from model.mnist_diffusion.UNet_advanced import UNetAdvanced
from model.schedulers import NoiseScheduler


# Function to generate samples using a trained diffusion model
def generate_samples(args, study):
    results_dir = args.results_dir
    # Construct path to saved model using best trial info from the Optuna study
    diff_model_path = f'trial_{study.best_trial.number}_losses_model_{study.best_trial.user_attrs["diff_model_name"]}'
    model_path = os.path.join(results_dir, diff_model_path)

    # Load datasets and data transformation functions
    train_dataset = joblib.load(os.path.join(results_dir, "train_dataset.pkl"))
    val_dataset = joblib.load(os.path.join(results_dir, "val_dataset.pkl"))
    test_dataset = joblib.load(os.path.join(results_dir, "test_dataset.pkl"))
    data_transform = joblib.load(os.path.join(results_dir, "transform.pkl"))
    data_inverse_transform = joblib.load(os.path.join(results_dir, "inverse_transform.pkl"))

    # Prepare test dataloader (not used here, but kept for optional visualization)
    testdata_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    # Set the label to generate (e.g., digit 2 in MNIST)
    # label = 2
    # label = torch.tensor(label)
    label = None

    with torch.no_grad():
        # Initialize U-Net model and noise scheduler from best trial's parameters
        #nn_model = MnistUNet(study.best_trial.user_attrs["model_nn_config"])
        nn_model = UNetAdvanced(**study.best_trial.user_attrs["model_nn_config"])
        noise_scheduler = NoiseScheduler(**study.best_trial.user_attrs["noise_scheduler_kwargs"])
        noise_scheduler.num_gen_timesteps = 50  # Limit generation steps for faster sampling

        # Wrap in the diffusion model
        diff_model = study.best_trial.user_attrs["diff_model_class"](nn_model, noise_scheduler)

        # Prepare optimizer if needed (based on non-frozen parameters)
        non_frozen_parameters = [p for p in diff_model.parameters() if p.requires_grad]
        optimizer = None
        if len(non_frozen_parameters) > 0:
            optimizer = study.best_trial.user_attrs["optimizer_class"](
                non_frozen_parameters,
                **study.best_trial.user_attrs["optimizer_kwargs"]
            )

        # Load checkpoint (handle CUDA availability)
        if not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            print("checkpoint with cuda")
            checkpoint = torch.load(model_path)

        # Load training/validation losses and best model weights
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        print("best loss: {}".format(np.min(train_loss)))
        print("best epoch: {}".format(np.argmin(train_loss)))

        # Plot training/validation loss curves
        plot_train_valid_loss(train_loss, valid_loss)

        # Load best model state
        diff_model.load_state_dict(checkpoint['best_model_state_dict'])
        diff_model.to("cuda")
        diff_model.eval()

        # Load optimizer state if applicable
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['best_optimizer_state_dict'])

        epoch = checkpoint['best_epoch']
        print("Best epoch ", epoch)

        # Sampling setup
        batch_size_sample = 1
        n_samples = 50  # Number of images to generate
        generated_samples = []
        torch.manual_seed(333)  # Set seed for reproducibility

        # Get the shape of input images from training data
        image_shape = np.squeeze(train_dataset[0][0]).shape

        # Loop to generate and plot samples
        for i in range(n_samples):
            inv_samples, orig_samples = diff_model.sample(
                batch_size=batch_size_sample,
                shape=image_shape,
                inverse_transform=data_inverse_transform,
                labels=label
            )
            plot_sample(np.squeeze(inv_samples.cpu().numpy()), title="generated image")


# Load Optuna study (contains best model configuration and results)
def load_study(results_dir):
    study = joblib.load(os.path.join(results_dir, "study.pkl"))
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    return study


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='results directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])

    # Load study and generate samples
    study = load_study(args.results_dir)
    generate_samples(args, study)
