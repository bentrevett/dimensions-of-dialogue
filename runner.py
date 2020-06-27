import subprocess
import random

while True:

    hid_dim = 128
    z_dim = 100
    v_dim = 1000 #random.choice([100, 1000])
    image_channels = 3 #random.choice([1, 3])
    image_size = 32
    init_std = 0.2
    batch_size = 128
    lr = random.choice([0.00002, 0.0002, 0.002])
    beta_1 = 0.5
    beta_2 = 0.999
    n_epochs = 10
    n_iters = random.choice([100, 250, 500])
    noise_mult = random.choice([0, 1.0, 3.0, 5.0, 7.0, 9.0])
    noise_inc_min = random.choice([1.0, -100])
    if noise_inc_min == -100:
        noise_inc_fac = 1.0
    else:
        noise_inc_fac = random.choice([1.1, 1.25, 1.5, 2.0])

    for seed in [0, 1, 1234, 1111]:

        start = f'python main.py '
        args_1 = f'--seed {seed} --hid_dim {hid_dim} --z_dim {z_dim} --v_dim {v_dim} '
        args_2 = f'--image_channels {image_channels} --image_size {image_size} '
        args_3 = f'--init_std {init_std} --batch_size {batch_size} --one_hot --lr {lr} --beta_1 {beta_1} '
        args_4 = f'--beta_2 {beta_2} --n_epochs {n_epochs} --n_iters {n_iters} --noise_mult {noise_mult} '
        args_5 = f'--noise_inc_min {noise_inc_min} --noise_inc_fac {noise_inc_fac}'
        arg = start + args_1 + args_2 + args_3 + args_4 +args_5
        subprocess.call(arg, shell = True)