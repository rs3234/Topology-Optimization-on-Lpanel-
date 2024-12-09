import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Set device for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Customized parameters for each problem
network_dict = {
    "model_type": 'MLP',
    "hidden_layers": 6,  # Set default values
    "neurons": 100,
    "seed": 1,
    "activation": 'TrainableReLU',
    "init_coeff": 1.0
}

optimizer_dict = {
    "weight_decay": 1e-5,
    "n_epochs_RPROP": 10000,
    "n_epochs_LBFGS": 0,
    "optim_rel_tol_pretrain": 1e-6,
    "optim_rel_tol": 5e-7
}

training_dict = {
    "save_model_every_n": 100
}

numr_dict = {
    "alpha_constraint": 'nonsmooth',
    "gradient_type": 'numerical'
}

PFF_model_dict = {
    "PFF_model": 'AT2',
    "se_split": 'volumetric',
    "tol_ir": 5e-3
}

mat_prop_dict = {
    "mat_E": 1.0,
    "mat_nu": 0.18,
    "w1": 1.0,
    "l0": 0.01
}

# Domain definition
domain_extrema = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
crack_dict = {
    "x_init": [0],
    "y_init": [0],
    "L_crack": [0],
    "angle_crack": [0]
}

# Prescribed incremental displacement
loading_angle = torch.tensor([np.pi / 2])
disp = np.concatenate((np.linspace(0.0, 0.5, 11), np.linspace(0.55, 1.4, 86)), axis=0)
disp = disp[1:]

# Domain discretization
coarse_mesh_file = "meshed_geom1.msh"  # Ensure this file is uploaded
fine_mesh_file = "meshed_geom2.msh"  # Ensure this file is uploaded

# Setting output directory
model_path = Path('hl_' + str(network_dict["hidden_layers"]) +
                  '_Neurons_' + str(network_dict["neurons"]) +
                  '_activation_' + network_dict["activation"] +
                  '_coeff_' + str(network_dict["init_coeff"]) +
                  '_Seed_' + str(network_dict["seed"]) +
                  '_PFFmodel_' + str(PFF_model_dict["PFF_model"]) +
                  '_gradient_' + str(numr_dict["gradient_type"]))
model_path.mkdir(parents=True, exist_ok=True)
trainedModel_path = model_path / Path('best_models/')
trainedModel_path.mkdir(parents=True, exist_ok=True)
intermediateModel_path = model_path / Path('intermediate_models/')
intermediateModel_path.mkdir(parents=True, exist_ok=True)

# Save model settings to a text file
with open(model_path / Path('model_settings.txt'), 'w') as file:
    file.write(f'hidden_layers: {network_dict["hidden_layers"]}\n')
    file.write(f'neurons: {network_dict["neurons"]}\n')
    file.write(f'seed: {network_dict["seed"]}\n')
    file.write(f'activation: {network_dict["activation"]}\n')
    file.write(f'coeff: {network_dict["init_coeff"]}\n')
    file.write(f'PFF_model: {PFF_model_dict["PFF_model"]}\n')
    file.write(f'se_split: {PFF_model_dict["se_split"]}\n')
    file.write(f'gradient_type: {numr_dict["gradient_type"]}\n')
    file.write(f'device: {device}\n')

# Logging loss to TensorBoard
writer = SummaryWriter(model_path / Path('TBruns'))
