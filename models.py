import torch
import torch.nn as nn
import torch.nn.functional as F


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
        h = torch.cat([z, x], dim=1)
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

    def forward(self, x):
        x = F.leaky_relu(self.conv1_1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = x.view(-1, self.hid_dim*4*4*4)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
