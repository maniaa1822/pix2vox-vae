# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume):
    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img

def get_volume_views_cv2(volume, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)