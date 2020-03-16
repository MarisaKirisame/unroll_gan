import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from data import gaussian_data_generator, noise_sampler
import utils
from model import Generator, Discriminator
from time import gmtime, strftime
import higher
import argparse

def d_loop(G, D, d_optimizer, criterion):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def d_unrolled_loop(G, D, d_optimizer, criterion, d_gen_input=None):

    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    # d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward(create_graph=True)
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def d_unrolled_loop_higher(G, D, d_optimizer, criterion, d_gen_input=None):

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_optimizer.step(d_loss)  # note that `step` must take `loss` as an argument!

    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def g_loop(G, D, g_optimizer, d_optimizer, criterion):
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    if config.unrolled_steps > 0:
        if config.use_higher:
            backup = copy.deepcopy(D)

            with higher.innerloop_ctx(D, d_optimizer) as (functional_D, diff_D_optimizer):
                for i in range(config.unrolled_steps):
                    d_unrolled_loop_higher(G, functional_D, diff_D_optimizer, criterion, d_gen_input=gen_input)

                g_optimizer.zero_grad()
                g_fake_data = G(gen_input)
                dg_fake_decision = functional_D(g_fake_data)
                target = torch.ones_like(dg_fake_decision).cuda()
                g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

            D.load(backup)
            del backup
        else:
            backup = copy.deepcopy(D)
            for i in range(config.unrolled_steps):
                d_unrolled_loop(G, D, d_optimizer, criterion, d_gen_input=gen_input)

            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            target = torch.ones_like(dg_fake_decision).cuda()
            g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
            D.load(backup)
            del backup

    else:
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data)
        target = torch.ones_like(dg_fake_decision).cuda()
        g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    return g_error.cpu().item()


def g_sample():
    with torch.no_grad():
        gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()
        g_fake_data = G(gen_input)
        return g_fake_data.cpu().numpy()


def load_config(name):
    import importlib
    config = importlib.import_module('configs.' + name )
    return config

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='unroll_0')
    args = argparser.parse_args()

    config = load_config(args.config)

    exp_dir = os.path.join('./experiments', "{}_{}".format(args.config, strftime("%Y-%m-%d_%H:%M:%S", gmtime())))
    os.makedirs(exp_dir, exist_ok=True)

    dset = gaussian_data_generator(config.seed)
    dset.random_distribution()
    utils.plot_lines(points=dset.p, title='Weight of each gaussian', path='{}/gaussian_weight.png'.format(exp_dir))

    sample_points = dset.sample(100)
    utils.plot_scatter(points=sample_points, centers=dset.centers, title='Sampled data points',
                       path='{}/samples.png'.format(exp_dir))

    prefix = "unrolled_steps-{}-prior_std-{:.2f}".format(config.unrolled_steps, np.std(dset.p))
    print("Save file with prefix", prefix)

    G = Generator(input_size=config.g_inp, hidden_size=config.g_hid, output_size=config.g_out).cuda()
    D = Discriminator(input_size=config.d_inp, hidden_size=config.d_hid, output_size=config.d_out).cuda()
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

    def binary_cross_entropy(x, y):
        loss = -(x.log() * y + (1 - x).log() * (1 - y))
        return loss.mean()
    criterion = binary_cross_entropy

    d_optimizer = optim.Adam(D.parameters(), lr=config.d_learning_rate, betas=config.optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=config.g_learning_rate, betas=config.optim_betas)

    samples = []
    for it in tqdm(range(config.num_iterations)):
        d_infos = []
        for d_index in range(config.d_steps):
            d_info = d_loop(G, D, d_optimizer, criterion)
            d_infos.append(d_info)
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos

        g_infos = []
        for g_index in range(config.g_steps):
            g_info = g_loop(G, D, g_optimizer, d_optimizer, criterion)
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos

        if it % config.log_interval == 0:
            g_fake_data = g_sample()
            samples.append(g_fake_data)
            utils.plot_scatter(points=g_fake_data, centers=dset.centers,
                               title='[{}] Iteration {}'.format(prefix, it), path='{}/samples_{}.png'.format(exp_dir, it))
            print(d_real_loss, d_fake_loss, g_loss)

    utils.plot_samples(samples, config.log_interval, config.unrolled_steps, path='{}/samples_{}.png'.format(exp_dir, 'final'))