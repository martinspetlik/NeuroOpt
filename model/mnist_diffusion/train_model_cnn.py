import os
import sys
import argparse
import joblib
import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, BruteForceSampler
import time
import yaml
import shutil
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset.mnist_diffusion.dataset_preprocessing import prepare_dataset
from model.mnist_diffusion.diffusion_model import DiffusionModel
from model.mnist_diffusion.UNet import MnistUNet
from model.mnist_diffusion.UNet_advanced import UNetAdvanced
from model.schedulers import NoiseScheduler
from torch.utils.data import DataLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"]=""


def validate(model, validation_loader, loss_fn=nn.MSELoss(), use_cuda=False):
    """
    Evaluate the model on a validation set.

    Parameters:
    - model: the PyTorch model to evaluate
    - validation_loader: DataLoader for validation data
    - loss_fn: loss function to use (default: Mean Squared Error)
    - use_cuda: whether to use GPU if available

    Returns:
    - avg_vloss: average validation loss over all batches
    """
    running_vloss = 0.0

    # Turn off gradient computation for validation
    with torch.no_grad():
        for i, samples in enumerate(validation_loader):
            images, conds = samples  # images and conditioning information (e.g. labels)

            # Move data to GPU if available and requested
            if torch.cuda.is_available() and use_cuda:
                images = images.cuda()
                conds = conds.cuda()

            # Forward pass: model returns (true noise, predicted noise)
            noise, predicted_noise = model((images, conds))

            # Compute loss between true and predicted noise
            vloss = loss_fn(noise, predicted_noise)

            # Accumulate total loss
            running_vloss += vloss.item()

        # Compute average loss across all batches
        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


def train_one_epoch(model, optimizer, train_loader, loss_fn=nn.MSELoss(), use_cuda=True):
    """
    Train the model for one epoch.

    Parameters:
    - model: the PyTorch model to train
    - optimizer: the optimizer used for updating model parameters
    - train_loader: DataLoader for training data
    - loss_fn: loss function to use (default: Mean Squared Error)
    - use_cuda: whether to use GPU if available

    Returns:
    - train_loss: average training loss over all batches
    """
    running_loss = 0.0

    for i, samples in enumerate(train_loader):
        images, conds = samples  # images and conditioning information (e.g. labels)

        # Move data to GPU if available and requested
        if torch.cuda.is_available() and use_cuda:
            images = images.cuda()
            conds = conds.cuda()

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Forward pass: model returns (true noise, predicted noise)
        noise, predicted_noise = model((images, conds))

        # Compute loss
        loss = loss_fn(noise, predicted_noise)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate training loss
        running_loss += loss.item()

    # Compute average loss across all batches
    train_loss = running_loss / (i + 1)

    return train_loss


def objective(trial, config, data_dir):
    ####
    # Noise scheduler config
    ####
    # Set default beta scheduler type and configuration
    beta_scheduler_type = "linear"
    scheduler_kwargs = {}
    if "beta_scheduler_type" in config:
        beta_scheduler_type = config["beta_scheduler_type"]
    if "scheduler_kwargs" in config:
        scheduler_kwargs = config["scheduler_kwargs"]

    # Define loss function
    loss_fn = nn.MSELoss()

    # Sanity checks to ensure config has necessary keys
    assert "n_train_samples" in config
    assert "n_test_samples" in config
    assert "val_samples_ratio" in config

    # Prepare training, validation, and test datasets
    train_set, validation_set, test_set = prepare_dataset(study, config, data_dir, serialize_path=output_dir)
    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set), len(test_set)))

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)

    #####################
    #  Initilize model  #
    #####################
    assert "model_nn_config" in config
    model_nn_config = config["model_nn_config"]

    # Select model architecture
    model_class_name = config.get("model_class_name", "MnistUNet")
    if model_class_name == "MnistUNet":
        model_class = MnistUNet
        nn_model = model_class(model_nn_config)
    elif model_class_name == "UNetAdvanced":
        model_class = UNetAdvanced
        nn_model = model_class(**model_nn_config)

    #######################
    ### Noise scheduler ###
    #######################
    noise_scheduler_kwargs = {
        'beta_scheduler_type': beta_scheduler_type,
        'num_timesteps': config["num_timesteps"],
        'scheduler_kwargs': scheduler_kwargs
    }
    noise_scheduler = NoiseScheduler(**noise_scheduler_kwargs)

    #######################
    ### Diffusion model ###
    #######################
    diff_model = DiffusionModel(nn_model, noise_scheduler).to(device)

    #########################
    # Initialize optimizer  #
    #########################
    lr = config["learning_rate"]
    optimizer_name = config.get("optimizer_name", "AdamW")
    optimizer_kwargs = {"lr": lr, "weight_decay": 0}

    # Filter parameters that require gradient updates
    non_frozen_parameters = [p for p in diff_model.parameters() if p.requires_grad]

    optimizer = None
    if len(non_frozen_parameters) > 0:
        optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs)

    print("optimizer ", optimizer)

    # Log metadata in the trial object for later inspection
    trial.set_user_attr("diff_model_class", diff_model.__class__)
    trial.set_user_attr("diff_model_name", diff_model._name)
    trial.set_user_attr("nn_model_class", nn_model.__class__)
    trial.set_user_attr("model_nn_config", model_nn_config)
    trial.set_user_attr("noise_scheduler_kwargs", noise_scheduler_kwargs)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("optimizer_kwargs", optimizer_kwargs)
    trial.set_user_attr("loss_fn", loss_fn)
    trial.set_user_attr("config", config)

    #########################
    #  Training Preparation #
    #########################
    best_validation_loss = 1_000_000.  # Initialize with a large number
    start_time = time.time()
    epoch_training_loss_list = []
    epoch_validation_loss_list = []
    best_epoch = 0
    model_state_dict = {}
    optimizer_state_dict = {}
    model_path = f'trial_{trial.number}_losses_model_{diff_model._name}'

    ############################
    # Learning rate scheduler #
    ############################
    learning_rate_scheduler = None
    if "learning_rate_scheduler" in config and optimizer is not None:
        scheduler_config = config["learning_rate_scheduler"]
        print("scheduler patience: {}, factor: {}".format(scheduler_config["patience"], scheduler_config["factor"]))
        if scheduler_config.get("class") == "ReduceLROnPlateau":
            learning_rate_scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                patience=scheduler_config["patience"],
                factor=scheduler_config["factor"]
            )
        else:
            learning_rate_scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"]
            )

    ##################
    #  Training loop #
    ##################
    for epoch in range(config["num_epochs"]):
        # Train model for one epoch
        diff_model.train(True)
        epoch_training_loss = train_one_epoch(diff_model, optimizer, train_loader, loss_fn=loss_fn, use_cuda=use_cuda)
        diff_model.train(False)

        # Validate model after training epoch
        epoch_validation_loss = validate(diff_model, validation_loader, loss_fn=loss_fn, use_cuda=use_cuda)

        # Step learning rate scheduler if applicable
        if learning_rate_scheduler is not None:
            learning_rate_scheduler.step(epoch_validation_loss)
            learning_rate_scheduler_state_dict = learning_rate_scheduler.state_dict()
            print("learning rate scheduler lr: {}".format(learning_rate_scheduler._last_lr))

        # Log current epoch losses
        epoch_training_loss_list.append(epoch_training_loss)
        epoch_validation_loss_list.append(epoch_validation_loss)

        print("epoch: {}, LOSS train: {}, val: {}".format(epoch, epoch_training_loss, epoch_validation_loss))

        # Save best model based on validation loss
        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_epoch = epoch
            print("best epoch ", best_epoch)

            model_state_dict = diff_model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            model_path_epoch = os.path.join(output_dir, model_path + f"_best_{epoch}")
            learning_rate_scheduler_state_dict = learning_rate_scheduler.state_dict()

            # Save model checkpoint
            torch.save({
                'best_epoch': best_epoch,
                'best_model_state_dict': model_state_dict,
                'best_optimizer_state_dict': optimizer_state_dict,
                'best_learning_rate_scheduler_state_dict': learning_rate_scheduler_state_dict,
                'train_loss': epoch_training_loss_list,
                'valid_loss': epoch_validation_loss_list,
                'training_time': time.time() - start_time,
            }, model_path_epoch)

        # Report validation loss to Optuna
        trial.report(epoch_validation_loss, epoch)

    ###
    # Save final model state (regardless of best validation loss)
    ###
    model_path = os.path.join(output_dir, model_path)
    torch.save({
        'best_epoch': best_epoch,
        'best_model_state_dict': model_state_dict,
        'best_optimizer_state_dict': optimizer_state_dict,
        'best_learning_rate_scheduler_state_dict': learning_rate_scheduler_state_dict,
        'train_loss': epoch_training_loss_list,
        'valid_loss': epoch_validation_loss_list,
        'training_time': time.time() - start_time,
    }, model_path)

    return best_validation_loss


def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    ###
    # Parse command line arguments
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='Path to config')  # Path to the configuration file (YAML/JSON)
    parser.add_argument('data_dir', help='Data directory')      # Path to the training/validation data
    parser.add_argument('output_dir', help='Output directory')  # Directory to store outputs/results
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")  # Optional flag to enable CUDA
    args = parser.parse_args(sys.argv[1:])

    # Assign parsed arguments to variables
    data_dir = args.data_dir
    output_dir = args.output_dir
    config = load_config(args.config_path)  # Load experiment configuration
    use_cuda = args.cuda

    # Set device to CUDA if available and requested; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    ###
    # Set random seed for reproducibility
    ###
    random_seed = config["random_seed"]
    torch.backends.cudnn.enabled = False  # Disable nondeterministic CuDNN algorithms for reproducibility
    torch.manual_seed(random_seed)        # Set PyTorch random seed

    # Update output directory with seed info and ensure it's clean
    output_dir = os.path.join(output_dir, f"seed_{random_seed}")
    if os.path.exists(output_dir):
        #shutil.rmtree(output_dir)  # Delete old directory if it exists
        # Alternatively, raise an error if overwriting is not desired
        raise IsADirectoryError(f"Results output dir {output_dir} already exists")
    os.mkdir(output_dir)  # Create fresh output directory

    ####
    # Initialize Optuna study
    ####
    sampler = TPESampler(seed=random_seed)  # Default sampler for hyperparameter optimization
    if "optuna_sampler_class" in config:
        if config["optuna_sampler_class"] == "BruteForceSampler":
            sampler = BruteForceSampler(seed=random_seed)  # Optionally use a custom sampler

    study = optuna.create_study(sampler=sampler, direction="minimize")  # Create study object

    # Define the objective function wrapper
    def obj_func(trial):
        return objective(trial, config, data_dir)

    # Run optimization (only 1 trial here; increase n_trials for full search)
    study.optimize(obj_func, n_trials=1)

    # ================================
    # Analyze Results
    # ================================

    # Get completed and pruned trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display basic statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Show best trial results
    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    ###
    # Save and display results
    ###
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Drop timing info
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only successful (non-pruned) trials
    df = df.drop('state', axis=1)                 # Drop state column
    df = df.sort_values('value')                  # Sort by objective value (assumed to be accuracy/loss)
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)  # Save to CSV

    # Print sorted dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    ###
    # Hyperparameter importance analysis
    ###
    try:
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)

        # Print sorted list of hyperparameter importances
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
    except Exception as e:
        print(str(e))  # Catch and report errors during importance calculation

    ###
    # Save the study object for future analysis or reuse
    ###
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))
