import numpy as np
from PIL import Image


def to_npy_file(np_array, model_name, img_size):
    numpy_array = np.array(np_array)
    label = "_labels"
    if len(numpy_array.shape) > 2:
        label = ""
    np.save(f'datafiles/{model_name}{label}_{img_size[0]}x{img_size[0]}.npy', numpy_array)

