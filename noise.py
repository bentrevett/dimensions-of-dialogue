import torch
import kornia


class NoiseChannel:
    def __init__(self, noise_std, shift_pct, rot_angle):
        self.noise_std = noise_std
        self.shift_pct = shift_pct
        self.rot_angle = rot_angle

    def apply(self, images):
        images = self.apply_noise(images, self.noise_std)
        images = self.apply_shift(images, self.shift_pct)
        images = self.apply_rotation(images, self.rot_angle)
        return images

    def apply_noise(self, images, std):
        noise = torch.randn(*images.shape) * std
        noise = noise.to(images.device)
        images = images + noise
        return images

    def apply_shift(self, images, shift):
        batch_size = images.shape[0]
        image_shape = images[0].shape
        average_image_dim = (image_shape[1] + image_shape[2]) / 2
        min_val = int(round(shift * average_image_dim * -1))
        max_val = int(round(shift * average_image_dim))
        shift_x, shift_y = torch.zeros(2).uniform_(min_val, max_val + 1).int()
        if shift_x != 0:
            zeros = torch.zeros(batch_size, image_shape[0], image_shape[1], abs(shift_x))
            zeros = zeros.to(images.device)
            if shift_x > 0:  # shift right
                chunk = images[:, :, :, :-shift_x]
                images = torch.cat((zeros, chunk), dim=3)
            else:  # shift left
                shift_x = abs(shift_x)
                chunk = images[:, :, :, shift_x:]
                images = torch.cat((chunk, zeros), dim=3)
        if shift_y != 0:
            zeros = torch.zeros(batch_size, image_shape[0], abs(shift_y), image_shape[2])
            zeros = zeros.to(images.device)
            if shift_y > 0:  # shift down
                chunk = images[:, :, :-shift_y]
                images = torch.cat((zeros, chunk), dim=2)
            else:  # shift up
                shift_y = abs(shift_y)
                chunk = images[:, :, shift_y:]
                images = torch.cat((chunk, zeros), dim=2)
        return images

    def apply_rotation(self, images, angle):
        batch_size = images.shape[0]
        angles = torch.zeros(batch_size).uniform_(-angle, angle)
        angles = angles.to(images.device)
        images = kornia.geometry.transform.rotate(images, angles)
        return images
