import numpy as np
from PIL import Image


def imgs_to_npy_file(np_array, file_name, img_size):
    numpy_array = np.array(np_array)
    np.save(f'datafiles/{file_name}_{img_size[0]}x{img_size[0]}.npy', numpy_array)


def labels_to_npy_file(label_array, file_name, img_size):
    numpy_array = np.array(label_array)
    np.save(f'datafiles/{file_name}_{img_size[0]}x{img_size[0]}.npy', numpy_array)
