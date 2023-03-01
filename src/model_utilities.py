import numpy as np
import pandas as pd

import process_data as data
import utilities as utils
import visualization as vis

def tf_model_predict(model, verbose: int=0) -> pd.DataFrame:
    """
    Predict the labels of the test dataset using the `model`.

    Arguments:
    ----
        `model` : keras model
            Trained model to run predictions on
        `verbose` : int
            Any value that is not 0 will enable verbosity

    Returns:
    ----
         A dataframe of the test data with prediction labels.
    """
    # Get the test data
    test_data = data.main("test")
    test_data = test_data.reshape(-1, 28*28).astype('float32')

    # Run the model
    preds = model.predict(test_data, verbose=verbose)
    
    # Take the 10 outputs and condence into one (using the max probability)
    pred_labels = np.argmax(preds, axis=1) # Class labels
    perc_labels = np.amax(preds, axis=1) # Percent certainty of the lables

    data_frame = pd.DataFrame(data=test_data, columns=[f"Pixel{i}" for i in range(28*28)])
    data_frame['pred_labels'] = pred_labels
    data_frame['perc_labels'] = perc_labels

    return data_frame

def tf_make_models_graphs(model, train_history, data_frame: pd.DataFrame,
                          model_name: str, num_obvs: int=3,
                          save=False, show=False) -> None:
    """
    Create plots for the model. Both save and show cannot be true.

    Arguments:
    ----
        `model` : keras model
            Trained model to run predictions on
        `train_history` : keras.history
            The training history of the model
        `data_frame` : pd.DataFrame
            The labeled data with percent certainty of the labels
            The labels column should be called `pred_labels`
            The percent certainty of the labels column should be called `perc_labels`
        `model_name` : str
            The name of the model
        `num_obvs` : int
            Number of digits to graph
        `save` : bool
            If save is true, the graphs will be saved under docs/model_name
        `show` : bool
            If show is true, the graphs will be shown as they are created
    """
    epochs = train_history.params['epochs']
    train_history = train_history.history

    loss_fig, acc_fig = vis.metrics_plots(epochs, train_history['loss'],
                                          train_history['val_loss'], train_history['accuracy'],
                                          train_history['val_accuracy'], show=show)

    digits_fig = vis.plot_digits_low_high_prob(model, data_frame, model_name,
                                               num_obvs=num_obvs, show=show)

    vis.save_figure(loss_fig, model_name, "loss")
    vis.save_figure(acc_fig, model_name, "accuracy")
    vis.save_figure(digits_fig, model_name, "digits")
