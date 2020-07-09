import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import random

import noise
import models

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--v_dim', type=int, default=1000)
parser.add_argument('--image_channels', type=int, default=3)
parser.add_argument('--init_std', type=float, default=0.02)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--beta_2', type=float, default=0.999)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--n_iters', type=int, default=400)
parser.add_argument('--noise_std', type=float, default=1.0)
parser.add_argument('--shift_pct', type=float, default=0.1)
parser.add_argument('--rot_angle', type=float, default=10)
parser.add_argument('--noise_inc_min', type=float, default=1.0)
parser.add_argument('--noise_inc_fac', type=float, default=1.5)
parser.add_argument('--shift_inc_min', type=float, default=1.0)
parser.add_argument('--shift_inc_fac', type=float, default=1.1)
parser.add_argument('--rot_inc_min', type=float, default=1.0)
parser.add_argument('--rot_inc_fac', type=float, default=1.2)
args = parser.parse_args()

assert args.hid_dim > 0
assert args.z_dim > 0
assert args.v_dim > 0
assert args.image_channels in [1, 3]
assert args.init_std > 0
assert args.batch_size > 0
assert args.lr > 0
assert args.beta_1 > 0
assert args.beta_2 > 0
assert args.n_epochs > 0
assert args.n_iters > 0
assert args.noise_std >= 0
assert args.shift_pct <= 1 and args.shift_pct >= 0

os.makedirs('run', exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

G = models.Generator(args.hid_dim, args.z_dim, args.v_dim, args.image_channels)
D = models.Discriminator(args.hid_dim, args.v_dim, args.image_channels)


def initialize_params(m):
    """
    initialize neural network parameters
    """
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean=0, std=args.init_std)
        m.bias.data.zero_()


G.apply(initialize_params)
D.apply(initialize_params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = G.to(device)
D = D.to(device)


def make_batch(batch_size, v_dim, fixed=False):
    """
    makes a batch of one-hot data
    label is the index of the 1 in the one-hot vector
    label is chosen between 1 and v_dim as 0 is reserved for noise message
    when fixed=True, examples have labels 1,2,3,4,5,... etc.
    """
    x = torch.zeros(batch_size, v_dim+1, 1, 1)
    y = torch.zeros(batch_size).long()
    for i in range(batch_size):
        v = (i % v_dim) + 1 if fixed else np.random.randint(1, v_dim)
        x[i, v] = 1.0
        y[i] = v
    return x, y


fixed_z = torch.randn(args.batch_size, args.z_dim, 1, 1)
fixed_x, _ = make_batch(args.batch_size, args.v_dim, fixed=True)

fixed_z = fixed_z.to(device)
fixed_x = fixed_x.to(device)

criterion = nn.CrossEntropyLoss()

G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))

noise_channel = noise.NoiseChannel(args.noise_std, args.shift_pct, args.rot_angle)


def train(G, D, G_optimizer, D_optimizer, v_dim, z_dim, n_iters, noise_channel, batch_size, device):
    """
    performs n_iters of parameter updates
    first train the discriminator on generated and fake images
    then train the generator via how well the discriminator classifies the generated images
    "fake images" are noise created using the same mean and std as the generated images
    """

    D_losses = []
    G_losses = []
    y_fake = torch.zeros(batch_size).long().to(device)

    for _ in tqdm(range(n_iters)):

        # get batch of generated data
        x, y = make_batch(batch_size, v_dim)
        x = x.to(device)
        y = y.to(device)

        # train discriminator on generated images
        D.zero_grad()
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        gen_images = G(z, x)
        gen_images = noise_channel.apply(gen_images)
        pred_gen = D(gen_images)
        D_gen_loss = criterion(pred_gen, y)

        # train discriminator on fake images (noise)
        gen_images_dist = distributions.normal.Normal(gen_images.mean(), gen_images.std())
        noise_images = gen_images_dist.sample(gen_images.shape).to(device)
        noise_images = noise_channel.apply(noise_images)
        pred_noise = D(noise_images)
        D_noise_loss = criterion(pred_noise, y_fake)

        # update discriminator parameters
        D_loss = D_gen_loss + D_noise_loss
        D_loss.backward()
        D_optimizer.step()
        D_losses.append(D_loss.item())

        # train generator
        x, y = make_batch(batch_size, v_dim)
        x = x.to(device)
        y = y.to(device)

        G.zero_grad()
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)

        gen_images = G(z, x)
        gen_images = noise_channel.apply(gen_images)
        pred_gen = D(gen_images)
        G_loss = criterion(pred_gen, y)
        G_loss.backward()
        G_optimizer.step()
        G_losses.append(G_loss.item())

    return np.mean(D_losses), np.mean(G_losses)


def normalize_image(image):
    """
    ensure image values are all between 0-1
    """
    
    image_min = image.min().item()
    image_max = image.max().item()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def save_images(G, image_channels, fixed_z, fixed_x, file_name, normalize=True):
    """
    generate images using the fixed z and x
    plot in a square figure
    save in runs/{epoch_number}.png
    """

    if image_channels == 1:
        cmap = 'gray'
    else:
        cmap = 'viridis'

    n_images = len(fixed_x)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    G.eval()
    gen_images = G(fixed_z, fixed_x)
    G.train()

    fig = plt.figure(figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        image = gen_images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).squeeze().detach().cpu().numpy(), cmap=cmap)
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(file_name)

    return gen_images


for epoch in range(args.n_epochs):

    D_loss, G_loss = train(G, D, G_optimizer, D_optimizer, args.v_dim, args.z_dim,
                           args.n_iters, noise_channel, args.batch_size, device)

    print(f'Epoch: {epoch+1}, Noise Std: {noise_channel.noise_std}, Shift Pct: {noise_channel.shift_pct}, Rot Angle: {noise_channel.rot_angle}')
    print(f'D Loss: {D_loss}')
    print(f'G Loss: {G_loss}')

    image_path = os.path.join('run', f'{epoch+1}')

    gen_images = save_images(G, args.image_channels, fixed_z, fixed_x, image_path)

    # for each of the types of noise, increase amount by "inc_fac" if loss is below "inc_min"
    # helps prevent overfitting

    if (D_loss + G_loss) < args.noise_inc_min:
        noise_channel.noise_std *= args.noise_inc_fac

    if (D_loss + G_loss) < args.shift_inc_min:
        noise_channel.shift_pct *= args.shift_inc_fac

    if (D_loss + G_loss) < args.rot_inc_min:
        noise_channel.rot_angle *= args.rot_inc_fac
