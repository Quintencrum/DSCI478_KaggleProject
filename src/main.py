import numpy as np

from models import logit
import model_utilities as model_utils
import process_data as data


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def run_logit(epochs, verbose=0):
    model, train_history = logit.main(epochs=epochs, verbose=verbose)
    preds = model_utils.tf_model_predict(model, verbose=verbose)

    model_utils.tf_make_models_graphs(model, train_history, preds,
                                      "Logistic Regression",
                                      show=bool(verbose), save=True)
    

if __name__ == "__main__":
    #a,b,c,d = data.main("train", overwrite=False)
    run_logit(epochs=20, verbose=0)
    
