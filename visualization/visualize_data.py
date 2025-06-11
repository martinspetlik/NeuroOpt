import numpy as np
import matplotlib.pyplot as plt


def plot_target_prediction(all_targets, all_predictions, title_prefix="", r2=[None, None, None],
                           nrmse=[None, None, None], x_labels=[r'$k_{xx}$', r'$k_{xy}$', r'$k_{yy}$']):
    all_targets = np.squeeze(all_targets)
    all_predictions = np.squeeze(all_predictions)

    print("len(all_targets.shape) ", len(all_targets.shape))

    if len(all_targets.shape) == 1:
        all_targets = np.reshape(all_targets, (*all_targets.shape, 1))
        all_predictions = np.reshape(all_predictions, (*all_predictions.shape, 1))

    n_channels = 1 if len(all_targets.shape) == 1 else all_targets.shape[1]

    #x_labels = [r'$k_{xx}$', r'$k_{xy}$', r'$k_{yy}$']
    titles = ['k_xx', 'k_xy', 'k_yy']

    for i in range(n_channels):
        k_target, k_predict = all_targets[:, i], all_predictions[:, i]
        plot_hist(k_target, k_predict, xlabel=x_labels[i], title=title_prefix + titles[i])
        plot_t_p(k_target, k_predict, label=x_labels[i], title=title_prefix + titles[i], r2=r2[i], nrmse=nrmse[i])

    # k_xx_target, k_xx_pred = all_targets[:, 0], all_predictions[:, 0]
    # k_xy_target, k_xy_pred = all_targets[:, 1], all_predictions[:, 1]
    # k_yy_target, k_yy_pred = all_targets[:, 2], all_predictions[:, 2]
    #
    # plot_hist(k_xx_target, k_xx_pred, xlabel=r'$k_{xx}$', title="k_xx")
    # plot_hist(k_xy_target, k_xy_pred, xlabel=r'$k_{xy}$', title="k_xy")
    # plot_hist(k_yy_target, k_yy_pred, xlabel=r'$k_{yy}$', title="k_yy")
    #
    # plot_t_p(k_xx_target, k_xx_pred, label=r'$k_{xx}$', title="k_xx")
    # plot_t_p(k_xy_target, k_xy_pred, label=r'$k_{xy}$', title="k_xy")
    # plot_t_p(k_yy_target, k_yy_pred, label=r'$k_{yy}$', title="k_yy")


def plot_hist(target, prediction, xlabel="k", title="hist"):
    plt.hist(target, bins=60,  color="red", label="target", density=True)
    plt.hist(prediction, bins=60, color="blue", alpha=0.5, label="prediction", density=True)
    plt.xlabel(xlabel)
    #plt.ylabel("Frequency for relative")
    plt.legend()
    plt.savefig("hist_" + title + ".pdf")
    plt.show()


def plot_t_p(targets, predictions, label="k", title="k", r2=None, nrmse=None):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 26})
    #matplotlib.rcParams.update({'font.size': 14})

    #a, b = np.polyfit(targets, predictions, 1)
    #print("lin a: {}, b: {}".format(a, b))
    # add points to plot
    #angle = np.arctan(a)
    #print("plot angle: {}".format(np.rad2deg(angle)))
    #print("plot angle: {}".format(angle))

    fig, ax = plt.subplots(1,1, figsize=(10, 10), dpi=600)
    ax.scatter(targets, predictions, s=15, alpha=0.2)#, edgecolors=(0, 0, 0))
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=4)

    print('len targets ', len(targets))
    print("len predictions ", len(predictions))

    if r2 is not None:
        plt.text(0.05, 0.95, r'$R^2 = {:.5f}$'.format(r2), transform=plt.gca().transAxes, ha='left', va='top')
    if nrmse is not None:
        plt.text(0.05, 0.89, r'$NRMSE = {:.5f}$'.format(nrmse), transform=plt.gca().transAxes, ha='left', va='top')

    # add line of best fit to plot
    #ax.plot(targets, a * targets + b, label="line fit")

    ax.set_xlabel('Targets')
    ax.set_ylabel('Predictions')
    plt.gca().ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    #ax.ticklabel_format(style='sci')
    plt.title(label)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(title + ".pdf")
    plt.show()

    #plt.savefig(title + ".pdf")


    # fig, ax = plt.subplots()
    # ax.scatter(targets, yr, edgecolors=(0, 0, 0))
    # ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=4)
    # # add line of best fit to plot
    # #ax.plot(targets, a * targets + b, label="line fit")
    #
    # ax.set_xlabel('Targets')
    # ax.set_ylabel('Rotated Predictions')
    # plt.title(label)
    # plt.legend()
    # #plt.savefig(title + ".png")
    # # plt.savefig(title + ".pdf")
    # plt.show()


def plot_train_valid_loss(train_loss, valid_loss):
    plt.plot(train_loss, label="train loss")
    plt.plot(valid_loss, label="valid loss")
    #plt.yscale("log")
    #plt.ylim([0, np.min([10000, np.max(train_loss), np.max(valid_loss)])])
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("train_val_loss.pdf")
    plt.show()


def plot_dataset(data_loader):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})

    targets_list = []
    for idx, data in enumerate(data_loader):
        input, targets = data
        # if len(targets.size()) > 1 and np.sum(targets.size()) / len(targets.size()) != 1:
        #     targets = torch.squeeze(targets.float())

        # print("targets ", targets)
        # print("type(targets) ", type(targets))
        # print("targeets shape ", targets.shape)

        targets = np.squeeze(targets.numpy())
        targets_list.append(targets)

    targets_arr = np.array(targets_list)
    print("target arr shape ", targets_arr.shape)

    n_channels = 3
    x_labels = [r'$log(k_{xx})$', r'$k_{xy}$', r'$log(k_{yy})$']
    titles = ['k_xx', 'k_xy', 'k_yy']
    print("targets arr ", targets_arr)
    for i in range(n_channels):
        if i == 1:
            plot_data = targets_arr[:, i]
        else:
            plot_data = np.log10(targets_arr[:, i])
        print("plot data shape ", plot_data.shape)
        plt.hist(plot_data, bins=60, label="target", density=True)
        plt.xlabel(x_labels[i])
        #plt.legend()
        plt.tight_layout()
        plt.savefig("hist_data_" + titles[i] + ".pdf")
        plt.show()
    exit()


def plot_samples(data_loader, n_samples=10):
    import matplotlib.pyplot as plt
    for idx, data in enumerate(data_loader):
        if idx > n_samples:
            break
        input, output = data
        print("input shape ", input.shape)
        #img = img / 2 + 0.5  # unnormalize
        #npimg = img.numpy()
        plt_input = input[0]
        plt_output = output[0]

        print("plt_input ", plt_input)

        fig, axes = plt.subplots(nrows=1, ncols=plt_input.shape[0], figsize=(10, 10))
        pcm = axes[0].matshow(plt_input[0])
        fig.colorbar(pcm, ax=axes[0])

        pcm = axes[1].matshow(plt_input[1])
        fig.colorbar(pcm, ax=axes[1])

        pcm = axes[2].matshow(plt_input[2])
        fig.colorbar(pcm, ax=axes[2])

        if plt_input.shape[0] > 3:
            pcm = axes[3].matshow(plt_input[3])
            fig.colorbar(pcm, ax=axes[3])

        #fig.colorbar(caxes)
        plt.savefig("input_{}.pdf".format(idx))
        plt.show()

        print("plt output shape ", plt_output.shape)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        pcm = axes.matshow(np.reshape(plt_output, (plt_output.shape[0], 1)))
        fig.colorbar(pcm, ax=axes)
        # fig.colorbar(caxes)
        plt.savefig("output_{}.pdf".format(idx))
        plt.show()


        #from metamodel.cnn.visualization.visualize_tensor import reshape_to_tensors, plot_cond_tn
        #cond_tn_target = reshape_to_tensors(plt_output, dim=2)[0:2, 0:2]

        #plot_cond_tn(cond_tn_target, label="target_tn_", color="red")


        # plt.matshow(plt_input[0])
        # plt.matshow(plt_input[1])
        # plt.matshow(plt_input[2])
        # plt.show()

    exit()
