import numpy as np

np.set_printoptions(suppress=True)

def unit_sticker(sticker_pixel,W,H,N):
    qN = np.zeros((N, 3))

    for i in range(len(sticker_pixel)):
        el = (np.pi / 2) - ((sticker_pixel[i][1] / H) * np.pi)
        az = ((2 * np.pi * (W - sticker_pixel[i][0])) / W) - np.pi
        qN[i] = (np.array([[(np.cos(el) * np.cos(az))],
                            [np.cos(el) * np.sin(az)],
                            [np.sin(el)]])).T


    return qN
