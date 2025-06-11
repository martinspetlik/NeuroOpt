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
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
# #from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
# from metamodel.cnn.visualization.visualize_data import plot_samples, plot_dataset
from dataset.mnist_diffusion.dataset_preprocessing import prepare_dataset
from model.mnist_diffusion.diffusion_model import DiffusionModel
from model.mnist_diffusion.UNet import MnistUNet
from model.schedulers import NoiseScheduler
from torch.utils.data import DataLoader


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"]=""


def validate(model, validation_loader, loss_fn=nn.MSELoss(), use_cuda=False):
    running_vloss = 0.0
    with torch.no_grad():
        for i, samples in enumerate(validation_loader):
            samples, cond = samples
            if torch.cuda.is_available() and use_cuda:
                samples = samples.cuda()

            noise, predicted_noise = model(samples)

            vloss = loss_fn(noise, predicted_noise)

            running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

    return avg_vloss


def train_one_epoch(model, optimizer, train_loader, loss_fn=nn.MSELoss(), use_cuda=True):
    """
    Train NN
    :param model:
    :param optimizer:
    :param loss_fn:
    :return:
    """
    running_loss = 0.
    for i, samples in enumerate(train_loader):

        samples, cond = samples

        if torch.cuda.is_available() and use_cuda:
            samples = samples.cuda()


        optimizer.zero_grad()

        noise, predicted_noise = model(samples)


        loss = loss_fn(noise, predicted_noise)

        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    train_loss = running_loss / (i + 1)
    return train_loss


def objective(trial, config, data_dir):

    ####
    # Noise scheduler config
    ####
    beta_scheduler_type = "linear"
    scheduler_kwargs = {}
    if "beta_scheduler_type" in config:
        beta_scheduler_type = config["beta_scheduler_type"]
    if "scheduler_kwargs" in config:
        scheduler_kwargs = config["scheduler_kwargs"]

    loss_fn = nn.MSELoss()

    assert "n_train_samples" in config
    assert "n_test_samples" in config
    assert "val_samples_ratio" in config

    train_set, validation_set, test_set = prepare_dataset(study, config, data_dir, serialize_path=output_dir)

    print("len(trainset): {}, len(valset): {}, len(testset): {}".format(len(train_set), len(validation_set), len(test_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["batch_size_train"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size_test"], shuffle=False)


    #####################
    #####################
    #  Initilize model #
    ####################
    ####################
    model_nn_config = {}
    assert "model_nn_config" in config
    model_nn_config = config["model_nn_config"]

    ####
    ## SimpleUNet
    ####
    model_class_name = "MnistUNet"
    if "model_class_name" in config:
        model_class_name = config["model_class_name"]

    if model_class_name == "MnistUNet":
        model_class = MnistUNet
    # elif model_class_name == "...":
    #     model_class = ...


    print("model nn config ", model_nn_config)

    nn_model = model_class(model_nn_config)

    #######################
    ### Noise scheduler ###
    #######################
    noise_scheduler_kwargs = {'beta_scheduler_type': beta_scheduler_type,
                              'num_timesteps': config["num_timesteps"],
                              'scheduler_kwargs': scheduler_kwargs}
    noise_scheduler = NoiseScheduler(**noise_scheduler_kwargs)

    #######################
    ### Diffusion model ###
    #######################
    diff_model = DiffusionModel(nn_model, noise_scheduler).to(device)

    #########################
    #########################
    #########################
    # Initialize optimizer
    lr = config["learning_rate"]
    optimizer_name = "AdamW"
    if "optimizer_name" in config:
        optimizer_name = config["optimizer_name"]

    optimizer_kwargs = {"lr": lr, "weight_decay": 0}
    non_frozen_parameters = [p for p in diff_model.parameters() if p.requires_grad]
    optimizer = None
    #print("optimizer kwargs ", optimizer_kwargs)

    #print("non frozen parameters ", non_frozen_parameters)
    if len(non_frozen_parameters) > 0:
        optimizer = getattr(optim, optimizer_name)(params=non_frozen_parameters, **optimizer_kwargs)

    print("optimizer ", optimizer)

    trial.set_user_attr("diff_model_class", diff_model.__class__)
    trial.set_user_attr("diff_model_name", diff_model._name)
    trial.set_user_attr("nn_model_class", nn_model.__class__)
    trial.set_user_attr("model_nn_config", model_nn_config)
    trial.set_user_attr("noise_scheduler_kwargs", noise_scheduler_kwargs)
    trial.set_user_attr("optimizer_class", optimizer.__class__)
    trial.set_user_attr("optimizer_kwargs", optimizer_kwargs)
    trial.set_user_attr("loss_fn", loss_fn)
    trial.set_user_attr("config", config)

    # Training of the model
    best_validation_loss = 1_000_000.
    start_time = time.time()
    epoch_training_loss_list = []
    epoch_validation_loss_list = []
    best_epoch = 0
    model_state_dict = {}
    optimizer_state_dict = {}

    model_path = 'trial_{}_losses_model_{}'.format(trial.number, diff_model._name)

    ############################
    # Learning rate scheduler ##
    ############################
    learning_rate_scheduler = None
    if "learning_rate_scheduler" in config and optimizer is not None:
        print("scheduler patience: {}, factor: {}".format(config["learning_rate_scheduler"]["patience"], config["learning_rate_scheduler"]["factor"]))
        if "class" in config["learning_rate_scheduler"]:
            if config["learning_rate_scheduler"]["class"] == "ReduceLROnPlateau":
                learning_rate_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=config["learning_rate_scheduler"]["patience"],
                                                           factor=config["learning_rate_scheduler"]["factor"])
            else:
                learning_rate_scheduler = lr_scheduler.StepLR(optimizer,
                                                              step_size=config["learning_rate_scheduler"]["step_size"],
                                                              gamma=config["learning_rate_scheduler"]["gamma"])

    ##################
    #  Training loop #
    ##################
    for epoch in range(config["num_epochs"]):
        ###
        # Training - one epoch
        ###
        diff_model.train(True)
        epoch_training_loss = train_one_epoch(diff_model, optimizer, train_loader, loss_fn=loss_fn, use_cuda=use_cuda)  # Train the model
        diff_model.train(False)

        ###
        # Validation step
        ###
        epoch_validation_loss = validate(diff_model, validation_loader, loss_fn=loss_fn, use_cuda=use_cuda)  # Evaluate the model

        ###
        # Adjust learning rate
        ###
        if learning_rate_scheduler is not None:
            learning_rate_scheduler.step(epoch_validation_loss)
            learning_rate_scheduler_state_dict = learning_rate_scheduler.state_dict()
            print("learning rate scheduler lr: {}".format(learning_rate_scheduler._last_lr))

        epoch_training_loss_list.append(epoch_training_loss)
        epoch_validation_loss_list.append(epoch_validation_loss)

        print("epoch: {}, LOSS train: {}, val: {}".format(epoch, epoch_training_loss, epoch_validation_loss))

        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_epoch = epoch
            print("best epoch ", best_epoch)


            ###
            # Save the best model so far
            ###
            model_state_dict = diff_model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            model_path_epoch = os.path.join(output_dir, model_path + "_best_{}".format(epoch))
            learning_rate_scheduler_state_dict = learning_rate_scheduler.state_dict()
            torch.save({
                'best_epoch': best_epoch,
                'best_model_state_dict': model_state_dict,
                'best_optimizer_state_dict': optimizer_state_dict,
                'best_learning_rate_scheduler_state_dict': learning_rate_scheduler_state_dict,
                'train_loss': epoch_training_loss_list,
                'valid_loss': epoch_validation_loss_list,
                'training_time': time.time() - start_time,
            }, model_path_epoch)

        trial.report(epoch_validation_loss, epoch)

    ###
    # Save the last model
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
    parser.add_argument('config_path', help='Path to config')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument("-c", "--cuda", default=False, action='store_true', help="use cuda")
    args = parser.parse_args(sys.argv[1:])
    data_dir = args.data_dir
    output_dir = args.output_dir
    config = load_config(args.config_path)
    use_cuda = args.cuda

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # Make runs repeatable
    random_seed = config["random_seed"]
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)
    output_dir = os.path.join(output_dir, "seed_{}".format(random_seed))
    if os.path.exists(output_dir):
        #shutil.rmtree(output_dir)
        raise IsADirectoryError("Results output dir {} already exists".format(output_dir))
    os.mkdir(output_dir)

    ####
    # Optuna auxiliary commands
    ####
    sampler = TPESampler(seed=random_seed)
    if "optuna_sampler_class" in config:
        if config["optuna_sampler_class"] == "BruteForceSampler":
            sampler = BruteForceSampler(seed=random_seed)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    def obj_func(trial):
        return objective(trial, config, data_dir)
    study.optimize(obj_func, n_trials=1)

    # ================================
    # Results
    # ================================
    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    try:
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)

        # Display the most important hyperparameters
        print('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
    except Exception as e:
        print(str(e))

    # serialize optuna study object
    joblib.dump(study, os.path.join(output_dir, "study.pkl"))