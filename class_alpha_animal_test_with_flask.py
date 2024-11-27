# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q diffusers

import torch
import os
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
# from dataset_loader import Dataset
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from torchvision.utils import save_image
import json
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the dataset
class ClassConditionedUnet_alpha(nn.Module):
  def __init__(self, num_classes=39, class_emb_size=4):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=28,           # the target image resolution
        in_channels=1 + class_emb_size, # Additional input channels for class cond.
        out_channels=1,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),
        down_block_types=(
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape

    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)

# Create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule='squaredcos_cap_v2')

#@markdown Training loop (10 Epochs):


# How many runs through the data should we do?
n_epochs = 10

# Our network
model_alpha = ClassConditionedUnet_alpha().to(device)

# Our loss function
loss_fn = nn.MSELoss()


# The optimizer
opt = torch.optim.Adam(model_alpha.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
losses = []


class ClassConditionedUnet_Animal(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=256,           # the target image resolution
        in_channels=3 + class_emb_size, # Additional input channels for class cond.
        out_channels=1,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),
        down_block_types=(
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape

    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)

# Create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule='squaredcos_cap_v2')

#@markdown Training loop (10 Epochs):

\
# How many runs through the data should we do?
n_epochs = 10

# Our network
model_Animal = ClassConditionedUnet_Animal().to(device)

# Our loss function
loss_fn = nn.MSELoss()


# The optimizer
opt = torch.optim.Adam(model_Animal.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
losses = []



#@markdown Sampling some different digits:

# Prepare random x to start from, plus some desired labels y
f = open('label_alphanumeric.json')
class_label_alphanumeric = json.load(f)
def predict_alphanumeric(key):
    value = class_label_alphanumeric.get(key)
    x = torch.randn(1, 1, 28, 28).to(device)
    y = torch.tensor([value]).flatten().to(device)

    model_alpha.load_state_dict(torch.load("Checkpoint/Checkpoint_2/model_v2.pt"))
    model_alpha.eval()
    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residue = model_alpha(x, t, y)
            # residual =   # Again, note that we pass in our labels y
        # Update sample with step
        x = noise_scheduler.step(residue, t, x).prev_sample

# Show the results
    x = x[0]
    # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')   
    save_image(x, f'static/Predicted_Alphanumeric_Image/{key}.png')
    path = os.path.join("static/Predicted_Alphanumeric_Image/",str(key)+".png")
    return path


f = open('label_animal.json')
class_label_animal = json.load(f)
def predict_animal(key):
    value = class_label_animal.get(key)
    x = torch.randn(1, 3, 128, 128).to(device)
    y = torch.tensor([value]).flatten().to(device)

    model_Animal.load_state_dict(torch.load("Checkpoint/Checkpoint_1/model_animal.pt"))
    model_Animal.eval()
    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residue = model_Animal(x, t, y)
            # residual =   # Again, note that we pass in our labels y
        # Update sample with step
        x = noise_scheduler.step(residue, t, x).prev_sample

# Show the results
    x = x[0]
    # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')   
    save_image(x, f'static/Predicted_Animal_Image/{key}.png')
    path = os.path.join("static/Predicted_Animal_Image/",str(key)+".png")
    return path


import io
from flask import Flask, send_file

app = Flask(__name__)

@app.route('/<key>')

def generate_image(key):
    
    for i in range(2):
        if(key in class_label_alphanumeric.keys()):
            path_var_alpha1 = predict_alphanumeric(key)
            path_var_alpha2 = predict_alphanumeric(key)
            with open(path_var_alpha1, 'rb') as f:
                contents = f.read()
            os.remove(path_var_alpha1)
            return send_file(
            io.BytesIO(contents),
            mimetype="image/png"
        )
        elif(key in class_label_animal.keys()):
            path_var_animal = predict_animal(key)
            return send_file(path_var_animal, mimetype='image/png')
        else:
            return "False"

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5001, debug=True, threaded=False)

