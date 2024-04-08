import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

import numpy as np

from diffusion_utilities import *


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  # assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)  # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)  # down2 #[10, 256, 4,  4]

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 8, self.h // 8),  # up-sample
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 8, self.h // 8),  # up-sample
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),  # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)  # [10, 256, 8, 8]
        down2 = self.down2(down1)  # [10, 256, 4, 4]

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


class BagOfWordsTextEncoder():
    def __init__(self):
        self.total_words = 0
        self.idx_to_word = {}
        self.word_to_idx = {}
        self.all_words = set()

    def fit(self, text_array):
        for sentence in text_array:
            words = [word.lower() for word in sentence.split()]
            for word in words:
                # Add all words. Since its a set, there won't be duplicates.
                self.all_words.add(word)

        for idx, word in enumerate(self.all_words):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            # Set the vocab size.
        self.total_words = len(self.all_words)

    def encode(self, text):
        if type(text) == str:
            encoded = self._transform_sentence(text.split())
        elif type(text) == list and type(text[0]) == str:
            encoded = np.empty((len(text), self.total_words))
            # Iterate over all sentences - this can be parallelized.
            for row, sentence in enumerate(text):
                # Substitute each row by the sentence BoW.
                encoded[row] = self._transform_sentence(sentence.split())
        else:
            raise TypeError(f"You must pass either a string or list of strings for transformation. type is {type(string)}")
        return encoded

    def _transform_sentence(self, list_of_words):
        transformed = np.zeros(self.total_words)
        for word in list_of_words:
            if word in self.all_words:
                word_idx = self.word_to_idx[word]
                transformed[word_idx] += 1
        return transformed


class TF_IDF:
    def __init__(self):
        self.tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

    def fit(self, text_array):
        self.tfidf_wm = self.tfidfvectorizer.fit_transform(text_array)
        self.tfidf_tokens = self.tfidfvectorizer.get_feature_names()

    def encode(self, text):
        df_tfidfvect = pd.DataFrame(data=self.tfidf_wm.toarray(), index=['Doc1', 'Doc2'], columns=self.tfidf_tokens)
        return df_tfidfvect.to_numpy()
