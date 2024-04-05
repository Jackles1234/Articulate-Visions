import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_utilities import *
from main import device, n_epoch, lrate, nn_model, optim, timesteps, save_dir, batch_size
from train_utils import perturb_input

# training with context code
# set into train mode

from datasets import load_dataset



def train_model():
    nn_model.train()

    dataset = load_dataset("poloclub/diffusiondb")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader, mininterval=2 )
        for x, c in pbar:   # x: images  c: context
            optim.zero_grad()
            x = x.to(device)
            c = c.to(x)

            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps, c=c)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        if ep%4==0 or ep == int(n_epoch-1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"context_model_{ep}.pth")
            print('saved model at ' + save_dir + f"context_model_{ep}.pth")

if __name__ == "__main__":
    train_model()