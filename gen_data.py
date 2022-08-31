import numpy as np
import utils


path = 'data/train/'
resolution = 2.5
utils.save_data(path, resolution)
utils.save_DTW_data(path)