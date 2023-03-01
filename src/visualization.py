import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import utilities as utils


def metrics_plots(epochs, train_losses, val_losses, train_accs, val_accs,
                       show: bool=False) -> (plt.figure, plt.figure):
    fig1 = plt.figure()
    plt.plot(range(epochs), train_losses, label = "Training loss")
    plt.plot(range(epochs), val_losses, label = "Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log loss")
    plt.legend()
    plt.title("Log loss vs training iterations")
    if show:
        plt.show()

    fig2 = plt.figure()
    plt.plot(range(epochs), train_accs, label = "Training accuracy")
    plt.plot(range(epochs), val_accs, label = "Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Validation iterations")
    if show:
        plt.show()

    return fig1, fig2

def plot_digits_low_high_prob(model, data_frame: pd.DataFrame,
                              model_name: str, num_obvs: int,
                              show: bool) -> plt.figure:
    """
    Plot the digits and model's classification of the test dataset.

    Arguments:
    ----
        `model` : keras model
            Trained model to run predictions on
        `data_frame` : pd.DataFrame
            The labeled data with percent certainty of the labels
            The labels column should be called `pred_labels`
            The percent certainty of the labels column should be called `perc_labels`
        `model_name` : str
            The name of the model
        `num_obvs` : int
            Number of min and max probabilities to visualize

    Returns:
    ----
        A figure of the test data against prediction labels.
    """
    data_frame = data_frame.sort_values('perc_labels')
    
    fig = plt.figure(figsize=(2, num_obvs))
    # grid for pairs of subplots
    grid = plt.GridSpec(2, 1)

    subtitles = ["Lowest Probability", "Highest Probability"]
    for i in range(2):
        # create fake subplot just to title set of subplots
        fake = fig.add_subplot(grid[i])
        fake.set_title(f'{subtitles[i]}\n', fontweight='semibold', size=14)
        fake.set_axis_off()

        # create subgrid for two subplots without space between them
        # <https://matplotlib.org/2.0.2/users/gridspec.html>
        gs = gridspec.GridSpecFromSubplotSpec(1, num_obvs, subplot_spec=grid[i])

        for j in range(num_obvs):
            ax = fig.add_subplot(gs[j])

            if i == 0:
                img = data_frame.iloc[j, :-2].values.reshape((28,28))
                label = data_frame['pred_labels'].iloc[j]
            elif i == 1:
                ind = -(j+1)
                img = data_frame.iloc[ind, :-2].values.reshape((28,28))
                label = data_frame['pred_labels'].iloc[ind]
            
            ax.imshow(img, cmap=plt.cm.binary)
            ax.set_title(f"Labeled: {label}")
            ax.set_axis_off()

    fig.patch.set_facecolor('white')
    fig.suptitle(f'{model_name} Predicitions', fontweight='bold', size=16)
    fig.tight_layout()
    fig.set_figheight(6)
    fig.set_figwidth(8)

    if show:
        plt.show()

    return fig

def plot_confusion_matrix():
    pass

def save_figure(fig: plt.figure, model_name: str, file_name: str) -> None:
    """
    Save the current figure under docs/model_name/file_name.png

    Arguments:
    ----
        `fig` : plt.figure
            The figure to save
        `model_name` : str
            Name of the model
        `file_name` : str
            The name of the file to be saved (without a file extension)
    """
    dir_path = utils.get_project_path().parent.joinpath('docs', model_name)
    try: 
        os.mkdir(dir_path) 
    except OSError as error: 
        pass
    file_path = dir_path.joinpath(f"{file_name}.png")
    
    fig = plt.figure(fig)
    plt.savefig(file_path)

    
