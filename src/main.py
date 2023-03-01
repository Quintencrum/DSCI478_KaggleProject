import numpy as np

import process_data as data
from models import logit
from models import svm

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


if __name__ == "__main__":
    #a,b,c,d = data.main("train", overwrite=False)
    #logit.main2()
    svm.main()
