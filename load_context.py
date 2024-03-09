import numpy as np
import torch

import matplotlib.pyplot as plt
from data.data_loader import diffusiondb_pixelart_label_encoder
from diffusion_utilities import plot_sample
from main import n_feat, n_cfeat, height, device, timesteps, b_t, a_t, ab_t, save_dir, context_datafile
from models import ContextUnet
from IPython.display import HTML

nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

nn_model.load_state_dict(torch.load(context_datafile, map_location=device))
nn_model.eval()
print("Loaded in Context Model")


@torch.no_grad()
def sample_ddpm_context(n_sample, context, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t, c=context)  # predict noise e_(x_t,t, ctx)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


def show_images(imgs, nrow=2):
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4, 2))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

    nn_model.load_state_dict(torch.load(context_datafile, map_location=device))
    nn_model.eval()
    print("Loaded in Context Model")

    ctx = torch.from_numpy(diffusiondb_pixelart_label_encoder().encode("apple")).float()
    samples, intermediate = sample_ddpm_context(2, ctx)
    show_images(samples)
    animation_ddpm_context = plot_sample(intermediate, 32, 4, save_dir, "ani_run", None, save=False)
    HTML(animation_ddpm_context.to_jshtml())
