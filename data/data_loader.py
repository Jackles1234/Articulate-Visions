import numpy as np
from datasets import load_dataset
try:
    from data_utils import to_npy_file
    from text_encoders import BagOfWordsTextEncoder
except ModuleNotFoundError:
    from data.data_utils import to_npy_file
    from data.text_encoders import BagOfWordsTextEncoder


text_encoder = BagOfWordsTextEncoder()


def polioclub_diffusiondb():
    """
    Dataset: https://huggingface.co/datasets/poloclub/diffusiondb
    uses the huggingface dataset library helper to use
    this is a massive dataset with labels that were used in stable diffusion
    """

    dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k', trust_remote_code=True)['train']
    images = dataset['image']
    labels = dataset['prompt']
    nsfw = dataset['image_nsfw']
    images = [item for item, condition in zip(images, nsfw) if condition < .1]
    labels = [item for item, condition in zip(labels, nsfw) if condition < .1]
    images = [np.array(image.convert("RGB").resize(img_size)) for image in images]
    # plt.imshow(images[0], interpolation='nearest')
    # plt.show()
    to_npy_file(images, "polioclub_diffusiondb", img_size)
    to_npy_file(text_encoder.encode(labels), "polioclub_diffusiondb", img_size)


def polioclub_diffusiondb_label_encoder():
    dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k', trust_remote_code=True)['train']
    labels = dataset['prompt']
    nsfw = dataset['image_nsfw']
    labels = [item for item, condition in zip(labels, nsfw) if condition < .1]
    text_encoder.fit(labels)
    return text_encoder


def nouns_sprite():
    # This function should work, but uses significant memory
    dataset = load_dataset("m1guelpf/nouns")['train']
    images = dataset['image']
    labels = dataset['text']
    images = [np.array(image.convert("RGB").resize(img_size)) for image in images]
    text_encoder.fit(labels)
    # plt.imshow(images[0], interpolation='nearest')
    # plt.show()
    to_npy_file(images, "nouns_sprite", img_size)
    to_npy_file(text_encoder.encode(labels), "nouns_sprite_labels", img_size)


def nouns_sprite_label_encoder():
    dataset = load_dataset("m1guelpf/nouns")["train"]
    labels = dataset['text']
    text_encoder.fit(labels)
    return text_encoder


def diffusiondb_pixelart():
    """
    Dataaset: https://huggingface.co/datasets/jainr3/diffusiondb-pixelart
    :return:
    """
    dataset = load_dataset("jainr3/diffusiondb-pixelart", trust_remote_code=True)['train']
    images = dataset['image']
    labels = dataset['text']
    text_encoder.fit(labels)
    images = [np.array(image.convert("RGB").resize(img_size)) for image in images]
    # plt.imshow(images[0], interpolation='nearest')
    # plt.show()
    to_npy_file(images, "diffusion_pixelart", img_size)
    to_npy_file(text_encoder.encode(labels), "diffusion_pixelart", img_size)


def diffusiondb_pixelart_label_encoder():
    dataset = load_dataset("jainr3/diffusiondb-pixelart")['train']
    labels = dataset['text']
    print(labels[0])
    text_encoder.fit(labels)
    return text_encoder


if __name__ == '__main__':
    img_size = (32, 32)
    diffusiondb_pixelart_label_encoder()
    # diffusiondb_pixelart()
    # polioclub_diffusiondb()
    # nouns_sprite()
