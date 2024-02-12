import numpy as np
from PIL import Image


def img_to_np_array(image_path):
    img = Image.open(image_path)
    np_img = np.array(img)


def create_npy_file(numpy_array, file_name):
    np.save(f'{file_name}.npy', numpy_array)
