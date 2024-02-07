
from models import ContextUnet


def model_init():
    # hyperparameters

    # diffusion hyperparameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat = 64 # 64 hidden dimension feature
    n_cfeat = 5 # context vector is of size 5
    height = 16 # 16x16 image
    save_dir = './weights/'

    # training hyperparameters
    batch_size = 100
    n_epoch = 32
    lrate=1e-3

    # construct DDPM noise schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
    ab_t[0] = 1

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
    # re setup optimizer
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
    return nn_model, optim



def main():
    nn_model, optim = model_init()

    # load in pretrain model weights and set to eval mode
    nn_model.load_state_dict(torch.load(f"{save_dir}/context_model_trained.pth", map_location=device))
    nn_model.eval() 
    print("Loaded in Context Model")

if __name__ == "__main__":
    main()
