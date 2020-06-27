import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions

import os
import random

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--v_dim', type=int, default=1000)
parser.add_argument('--image_channels', type=int, default=3)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--init_std', type=float, default=0.02)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--one_hot', action='store_true')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--beta_2', type=float, default=0.999)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--n_iters', type=int, default=400)
parser.add_argument('--noise_mult', type=float, default=5.0)
parser.add_argument('--noise_inc_min', type=float, default=1.0)
parser.add_argument('--noise_inc_fac', type=float, default=2.0)
args = parser.parse_args()

assert args.hid_dim > 0
assert args.z_dim > 0 
assert args.v_dim > 0
assert args.image_channels in [1, 3]
assert args.image_size > 0
assert args.init_std > 0
assert args.batch_size > 0
assert args.one_hot, 'Must use --one_hot currently!'
assert args.lr > 0
assert args.beta_1 > 0
assert args.beta_2 > 0
assert args.n_epochs > 0
assert args.noise_inc_min > 0 or args.noise_inc_min == -100
assert args.noise_mult >= 0
assert args.noise_inc_fac >= 1.0



run_name = '-'.join([f'{k}={v}' for k, v in vars(args).items()])

print(run_name)

run_folder = os.path.join('runs', run_name)

os.makedirs(run_folder)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

class Generator(nn.Module):
    def __init__(self, hid_dim, z_dim, v_dim, image_channels):
        super().__init__()
        self.deconv1_1 = nn.ConvTranspose2d(z_dim, hid_dim*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(hid_dim * 2)
        self.deconv1_2 = nn.ConvTranspose2d(v_dim + 1, hid_dim*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(hid_dim*2)
        self.deconv2 = nn.ConvTranspose2d(hid_dim*4, hid_dim*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(hid_dim*2)
        self.deconv3 = nn.ConvTranspose2d(hid_dim*2, hid_dim, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(hid_dim)
        self.deconv4 = nn.ConvTranspose2d(hid_dim, image_channels, 4, 2, 1)

    def forward(self, z, x):
        z = F.relu(self.deconv1_1_bn(self.deconv1_1(z)))     
        x = F.relu(self.deconv1_2_bn(self.deconv1_2(x)))
        h = torch.cat([z, x], dim = 1)
        h = F.relu(self.deconv2_bn(self.deconv2(h)))
        h = F.relu(self.deconv3_bn(self.deconv3(h)))
        h = torch.tanh(self.deconv4(h))
        return h

class Discriminator(nn.Module):
    def __init__(self, hid_dim, v_dim, image_channels):
        super().__init__()
        self.hid_dim = hid_dim
        self.conv1_1 = nn.Conv2d(image_channels, hid_dim//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(hid_dim//2, hid_dim*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(hid_dim*2)
        self.conv3 = nn.Conv2d(hid_dim*2, hid_dim*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(hid_dim*4)
        self.fc1 = nn.Linear(hid_dim*4*4*4, 512)
        self.fc2 = nn.Linear(512, v_dim+1)

    # forward method
    def forward(self, x, noise_mult = 0.0):
        
        x = F.leaky_relu(self.conv1_1(x), 0.2)        
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = x.view(-1, self.hid_dim*4*4*4)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

G = Generator(args.hid_dim, args.z_dim, args.v_dim, args.image_channels)
D = Discriminator(args.hid_dim, args.v_dim, args.image_channels)

def initialize_params(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean = 0, std = args.init_std)
        m.bias.data.zero_()

G.apply(initialize_params)
D.apply(initialize_params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = G.to(device)
D = D.to(device)

def make_batch(batch_size, v_dim, one_hot = True, fixed = False):
    x = torch.zeros(batch_size, v_dim+1, 1, 1)
    y = torch.zeros(batch_size).long()
    if one_hot:
        for i in range(batch_size):
            v = i % v_dim if fixed else np.random.randint(1, v_dim)
            x[i, v] = 1.0
            y[i] = v
    else:
        raise NotImplementedError
        for i in range(batch_size):
            v = i if fixed else np.random.randint(1, 2 ** v_dim)
            y[i] = v
            v = [int(i) for i in np.binary_repr(v, width=v_dim+1)]
            v = torch.FloatTensor(v).view(-1, 1, 1)
            x[i] = v
    return x, y

fixed_z = torch.randn(args.batch_size, args.z_dim, 1, 1)
fixed_x, _ = make_batch(args.batch_size, args.v_dim, one_hot = args.one_hot, fixed = True)

fixed_z = fixed_z.to(device)
fixed_x = fixed_x.to(device)

criterion = nn.CrossEntropyLoss()

G_optimizer = optim.Adam(G.parameters(), lr = args.lr, betas = (args.beta_1, args.beta_2))
D_optimizer = optim.Adam(D.parameters(), lr = args.lr, betas = (args.beta_1, args.beta_2))

def train(G, D, G_optimizer, D_optimizer, v_dim, z_dim, n_iters, noise_mult, batch_size, one_hot, device):

    D_losses = []
    G_losses = []
    y_fake = torch.zeros(batch_size).long().to(device)

    for i in tqdm(range(n_iters)):

        #get batch of generated data
        x, y = make_batch(batch_size, v_dim, one_hot = one_hot)
        x = x.to(device)
        y = y.to(device)

        #train discriminator on generated images
        D.zero_grad()
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        gen_images = G(z, x)
        noise = torch.randn(*gen_images.shape).to(gen_images.device) * noise_mult
        #gen_images = gen_images + noise
        pred_gen = D(gen_images + noise)
        #pred_gen = D(gen_images, noise_mult)
        D_gen_loss = criterion(pred_gen, y)

        #train discriminator on fake images (noise)
        noise = distributions.normal.Normal(gen_images.mean(), gen_images.std())
        noise = noise.sample(gen_images.shape).to(device)
        pred_noise = D(noise) #w/ no noise added
        D_noise_loss = criterion(pred_noise, y_fake)

        #update discriminator parameters
        D_loss = D_gen_loss + D_noise_loss
        D_loss.backward()
        D_optimizer.step()
        D_losses.append(D_loss.item())

        #train generator
        x, y = make_batch(batch_size, v_dim, one_hot = one_hot)
        x = x.to(device)
        y = y.to(device)
        
        G.zero_grad()
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)

        gen_images = G(z, x)
        noise = torch.randn(*gen_images.shape).to(gen_images.device) * noise_mult
        gen_images = gen_images + noise
        pred_gen = D(gen_images)
        #pred_gen = D(gen_images, noise_mult)
        G_loss = criterion(pred_gen, y)
        G_loss.backward()
        G_optimizer.step()
        G_losses.append(G_loss.item())

    return np.mean(D_losses), np.mean(G_losses)

def normalize_image(image):
    image_min = image.min().item()
    image_max = image.max().item()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def save_images(G, image_channels, image_size, fixed_z, fixed_x, file_name, normalize = True):
    
    n_images = len(fixed_x)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    G.eval()
    gen_images = G(fixed_z, fixed_x)
    G.train()

    fig = plt.figure(figsize = (10, 10))
    fig = plt.figure(figsize = (10, 10))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = gen_images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).squeeze().detach().cpu().numpy())
        ax.axis('off')

    fig.savefig(file_name)

    return gen_images

log_name  = os.path.join(run_folder, 'log.txt')

with open(log_name, 'w+') as f:
    f.write('D_loss\tG_loss\n')

for epoch in range(args.n_epochs):

    D_loss, G_loss = train(G, D, G_optimizer, D_optimizer, args.v_dim, args.z_dim, args.n_iters, args.noise_mult, args.batch_size, args.one_hot, device)
    print(f'Epoch: {epoch+1}, Noise Mult: {args.noise_mult}')
    print(f'D Loss: {D_loss}')
    print(f'G Loss: {G_loss}')

    with open(log_name, 'a+') as f:
        f.write(f'{D_loss}\t{G_loss}\n')

    image_name = os.path.join(run_folder, f'{epoch}')

    gen_images = save_images(G, args.image_channels, args.image_size, fixed_z, fixed_x, image_name)

    if D_loss + G_loss < args.noise_inc_min:
        args.noise_mult *= args.noise_inc_fac