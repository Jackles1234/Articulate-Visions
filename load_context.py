import numpy as np
import torch

import matplotlib.pyplot as plt
from data.data_loader import diffusiondb_pixelart_label_encoder
from diffusion_utilities import plot_sample
from main import generate_context, Context
# from main import n_feat, n_cfeat, height, device, timesteps, b_t, a_t, ab_t, save_dir, context_datafile
from models import ContextUnet
from IPython.display import HTML


@torch.no_grad()
def sample_ddpm_context(n_sample, context, training_context: Context, nn_model, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, training_context.height, training_context.height).to(training_context.device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(training_context.timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / training_context.timesteps])[:, None, None, None].to(training_context.device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t, c=context)  # predict noise e_(x_t,t, ctx)
        samples = denoise_add_noise(samples, i, eps, training_context, z)
        if i % save_rate == 0 or i == training_context.timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate


def denoise_add_noise(x, t, pred_noise, training_context: Context, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = training_context.b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - training_context.a_t[t])
                              / (1 - training_context.ab_t[t]).sqrt())) / training_context.a_t[t].sqrt()
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


def load_context(training_context: Context):
    nn_model = ContextUnet(
        in_channels=3,
        n_feat=training_context.n_feat,
        n_cfeat=training_context.n_cfeat,
        height=training_context.height
    ).to(training_context.device)

    nn_model.load_state_dict(
        torch.load("data/datafiles/diffusion_pixelartdiffusiondb_pixelart_(200, 256, 50).pth", map_location=training_context.device))
    nn_model.eval()
    print("Loaded in Context Model")

    ctx = torch.from_numpy(diffusiondb_pixelart_label_encoder().encode("doom eternal, game concept art, veins and "
                                                                       "worms, muscular, crustacean exoskeleton, "
                                                                       "chiroptera head, chiroptera ears, mecha, "
                                                                       "ferocious, fierce, hyperrealism, "
                                                                       "fine details, artstation, cgsociety, zbrush, "
                                                                       "no background ")).float()
    return sample_ddpm_context(2, ctx, training_context, nn_model)


if __name__ == "__main__":
    dataset = 'diffusiondb_pixelart'
    hyperparameters = {
        # 'timesteps': 300,
        'batch_size': 200,
        'n_feat': 256,
        'n_epoch': 50
    }
    context = generate_context(dataset, hyperparameters)

    samples, intermediate = load_context(context)
    x = intermediate[-1]
    x = np.moveaxis(x, 1, -1)  # change to Numpy image format (h,w,channels) vs (channels,h,w)
    # x = np.transpose(x, (1, 2, 0))
    # show_images(x)
    plt.figure(figsize=(2, 2))
    plt.imshow((x[0] * 255).astype(np.uint8))
    plt.show()
    # show_images(samples)
    # animation_ddpm_context = plot_sample(intermediate, 32, 4, context.save_dir, "ani_run", None, save=False)
    # HTML(animation_ddpm_context.to_jshtml())
