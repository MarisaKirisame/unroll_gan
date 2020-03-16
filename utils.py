import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn

def plot_lines(points, title, path):

    plt.plot(points)
    plt.title(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_scatter(points, centers, title, path):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig(path)
    plt.close()


def plot_samples(samples, log_interval, unrolled_steps, path):
    xmax = 5
    cols = len(samples)
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2 * cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(samps[:, 0], samps[:, 1], shaded=True, cmap='Greens', n_levels=20,
                              clip=[[-xmax, xmax]] * 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d' % (i * log_interval))

    ax.set_ylabel('%d unrolling steps' % unrolled_steps)
    plt.gcf().tight_layout()
    plt.savefig(path)
    plt.close()