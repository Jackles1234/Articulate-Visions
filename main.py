from data import data_loader
from diffusion_utilities import *
from models import ContextUnet


class Context:
    def __init__(self):
        self.save_dir = './weights/'
        self.model_name = "diffusion_pixelart"
        self.context_datafile = f"data/datafiles/{self.model_name}"
        self.images_path = "data/datafiles/sprites_1788_16x16.npy"
        self.labels_path = "data/datafiles/sprite_labels_nc_1788_16x16.npy"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
        # self.context_datafile = f"{save_dir}/{model_name}"

        self.timesteps = 100
        self.beta1 = 1e-4
        self.beta2 = 0.02
        self.n_feat = 64  # 64 hidden dimension feature
        self.n_cfeat = 1000  # context vector is of size 3115
        self.height = 16  # 16x16 image
        self.batch_size = 200
        self.n_epoch = 5
        self.lrate = 1e-3
        self.b_t = ((self.beta2 - self.beta1)
                    * torch.linspace(0, 1, self.timesteps + 1, device=self.device)
                    + self.beta1)
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

        self.nn_model = (ContextUnet(in_channels=3, n_feat=self.n_feat, n_cfeat=self.n_cfeat, height=self.height)
                         .to(self.device))
        self.optim = torch.optim.Adam(self.nn_model.parameters(), lr=self.lrate)

        self.hyperparameters = dict()
        self.save_path = f"{self.model_name}_{*self.hyperparameters,}.pth"

    def update_parameters(self, parameters: dict):
        for key, item in parameters.items():
            setattr(self, key, item)

    def set_hyperparameters(self, parameters: dict):
        self.hyperparameters = parameters

    def __str__(self):
        return f"Model: {self.model_name} \n With Parameters: {*list(self.hyperparameters.values()),}"


def generate_context(dataset, hyperparameters):
    training_context = Context()

    # Set context features
    context_encoder = getattr(data_loader, f"{dataset}_label_encoder")
    training_context.model_name = dataset
    training_context.n_cfeat = context_encoder().total_words
    # training_context.images_path = (f"data/datafiles/{training_context.model_name}"
    #                                 f"_{training_context.height}x{training_context.height}.npy")
    # training_context.labels_path = (f"data/datafiles/{training_context.model_name}_labels_"
    #                                 f"{training_context.height}x{training_context.height}.npy")

    # Set hyperparameters
    training_context.update_parameters(hyperparameters)
    return training_context