import itertools

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import data.data_loader as data_loader
from diffusion_utilities import *
from main import Context, generate_context
from pathlib import Path


def train_model(training_context: Context):
    training_context.nn_model.train()

    dataset = CustomDataset(training_context.images_path, training_context.labels_path, transform, null_context=False)
    dataloader = DataLoader(dataset, batch_size=training_context.batch_size, shuffle=True, num_workers=1)
    optim = torch.optim.Adam(training_context.nn_model.parameters(), lr=training_context.lrate)

    for ep in range(training_context.n_epoch):
        print(f'\nepoch {ep}\n')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = training_context.lrate * (1 - ep / training_context.n_epoch)

        pbar = tqdm(dataloader, mininterval=2)
        for x, c in pbar:  # x: images  c: context
            optim.zero_grad()
            x = x.to(training_context.device)
            c = c.to(x)

            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(training_context.device)
            c = c * context_mask.unsqueeze(-1)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, training_context.timesteps + 1, (x.shape[0],)).to(training_context.device)

            x_pert = training_context.ab_t.sqrt()[t, None, None, None] * x + (
                        1 - training_context.ab_t[t, None, None, None]) * noise

            # use network to recover noise
            pred_noise = training_context.nn_model(x_pert, t / training_context.timesteps, c=c)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        if ep % 4 == 0 or ep == int(training_context.n_epoch - 1):
            Path(training_context.context_datafile).mkdir(parents=True, exist_ok=True)

            torch.save(training_context.nn_model.state_dict(), training_context.context_datafile
                       + f"{training_context.model_name}_{*list(training_context.hyperparameters.values()),}.pth")

            print('saved model at ' + training_context.context_datafile + "/"
                  + f"{training_context.model_name}_{*list(training_context.hyperparameters.values()),}.pth")

def train_all_models():
    datasets = ['diffusiondb_pixelart', 'polioclub']

    """
    Set how the hyperparameters will train
    name of hyperparameter : (Start, Stop, Step)
    Must be a range
    """
    hyperparameters_iteration = {
        'timesteps': range(100, 300, 10),
        'batch_size': range(200, 300, 10),
        'n_feat': range(128, 300, 16),
        'n_epoch': range(50, 300, 10)
    }

    keys = hyperparameters_iteration.keys()
    values = hyperparameters_iteration.values()
    combinations = list(itertools.product(*values))

    # Convert each combination of values into a dictionary with the corresponding hyperparameter names
    combination_dicts = [{key: value for key, value in zip(keys, combination)} for combination in combinations]
    print(f"Total combinations: {len(combination_dicts)}\n")

    for dataset in datasets:
        for idx, combination in enumerate(combination_dicts):
            context = generate_context(dataset, combination)
            print(context, f"{idx}/{len(combination_dicts)}")

            train_model(context)


def train_one_model():
    dataset = 'diffusiondb_pixelart'
    hyperparameters = {
        # 'timesteps': 300,
        'batch_size': 200,
        'n_feat': 64,
        'n_epoch': 50
    }
    context = generate_context(dataset, hyperparameters)
    context.set_hyperparameters(hyperparameters)
    print(context)
    train_model(context)


if __name__ == "__main__":
    train_one_model()
