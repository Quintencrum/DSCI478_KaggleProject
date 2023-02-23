import numpy as np

from models import logit
import process_data as data


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


if __name__ == "__main__":
    a,b,c,d = data.main("train", overwrite=False)
    #logit.main2()
