from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import os
import re
import urllib

import PIL.Image

import datasets
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from datasets import load_dataset
from PIL import Image
import numpy as np

from data.data_utils import imgs_to_npy_file, labels_to_npy_file
from models import BagOfWordsTextEncoder

text_encoder = BagOfWordsTextEncoder()


def use_red_caps_dataset():
    """
    dataset: https://huggingface.co/datasets/red_caps
    images like
    # https://i.imgur.com/xJgfd.jpg
    # https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Filpvfpr8ex721.jpg
    This dataset is cumbersome due to it being real photos with strange text prompts,
    it is also cumbersome due to the dataset storing images on third pary sites that need to be hit
    to get the info
    :return:
    """
    USER_AGENT = get_datasets_user_agent()

    def fetch_single_image(image_url, timeout=None, retries=0):
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = PIL.Image.open(io.BytesIO(req.read()))
                break
            except Exception:
                image = None
        return image

    def fetch_images(batch, num_threads, timeout=None, retries=0):
        fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch["image"] = list(
                executor.map(lambda image_urls: [fetch_single_image_with_args(image_url) for image_url in image_urls],
                             batch["image_url"]))
        return batch

    def process_image_urls(batch):
        processed_batch_image_urls = []
        for image_url in batch["image_url"]:
            processed_example_image_urls = []
            image_url_splits = re.findall(r"http\S+", image_url)
            for image_url_split in image_url_splits:
                if "imgur" in image_url_split and "," in image_url_split:
                    for image_url_part in image_url_split.split(","):
                        if not image_url_part:
                            continue
                        image_url_part = image_url_part.strip()
                        root, ext = os.path.splitext(image_url_part)
                        if not root.startswith("http"):
                            root = "http://i.imgur.com/" + root
                        root = root.split("#")[0]
                        if not ext:
                            ext = ".jpg"
                        ext = re.split(r"[?%]", ext)[0]
                        image_url_part = root + ext
                        processed_example_image_urls.append(image_url_part)
                else:
                    processed_example_image_urls.append(image_url_split)
            processed_batch_image_urls.append(processed_example_image_urls)
        batch["image_url"] = processed_batch_image_urls
        return batch

    dset = load_dataset("red_caps", "rabbits_2017", trust_remote_code=True)
    dset = dset.map(process_image_urls, batched=True, num_proc=4)
    features = dset["train"].features.copy()
    features["image"] = datasets.Sequence(datasets.Image())
    num_threads = 20
    dset = dset.map(fetch_images, batched=True, batch_size=100, features=features,
                    fn_kwargs={"num_threads": num_threads})


def polioclub_diffusiondb():
    """
    Dataset: https://huggingface.co/datasets/poloclub/diffusiondb
    uses the huggingface dataset library helper to use
    this is a massive dataset with labels that were used in stable diffusion
    :return: 
    """

    dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')


def ms_coco_db():
    """
    Used by DALL-E to test, this is an external dataset that does not directly support python
    Coco dataset https://cocodataset.org/#download
    Python API: https://github.com/cocodataset/cocoapi
    The challenge with this dataset will be using makefiles to download and use the dataset
    No code with this one as it requires makefile dependencies
    :return:
    """
    pass


def diffusiondb_pixelart():
    """
    Dataaset: https://huggingface.co/datasets/jainr3/diffusiondb-pixelart
    :return:
    """
    img_size = (32, 32)
    dataset = load_dataset("jainr3/diffusiondb-pixelart", trust_remote_code=True)['train']
    images = dataset['image']
    labels = dataset['text']
    text_encoder.fit(labels)
    images = [np.array(image.convert("RGB").resize(img_size)) for image in images]
    imgs_to_npy_file(images, "diffusion_pixelart_db_img", img_size)
    labels_to_npy_file(text_encoder.encode(labels), "diffusion_pixelart_db_labels", img_size)


def diffusiondb_pixelart_label_encoder():
    dataset = load_dataset("jainr3/diffusiondb-pixelart")['train']
    labels = dataset['text']
    text_encoder.fit(labels)
    return text_encoder


if __name__ == '__main__':
    diffusiondb_pixelart()
