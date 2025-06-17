import numpy as np
import matplotlib.pyplot as plt


def plot_train_valid_loss(train_loss, valid_loss):
    plt.plot(train_loss, label="train loss")
    plt.plot(valid_loss, label="valid loss")
    #plt.yscale("log")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("train_val_loss.pdf")
    plt.show()


# Function to plot a single grayscale image, optionally saving it to a PDF
def plot_sample(image, title="MNIST sample", file_name=None):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Hide axis ticks and labels

    if file_name is not None:
        plt.savefig("{}.pdf".format(file_name), bbox_inches='tight')
    plt.show()
