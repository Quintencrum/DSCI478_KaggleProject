import numpy as np

from models import *
import process_data as data
import drawing_images as di


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


if __name__ == "__main__":
    # a,b,c,d = data.main("train", overwrite=False)
    #logit.main2()
    di.draw_one_to_nine()