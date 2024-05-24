import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import scipy.linalg
from matplotlib.transforms import Affine2D

from garnet.plots.base import BasePlot


class SlicePlot(BasePlot):
    def __init__(self, UB, W):
        self.fig, self.ax = plt.subplots(1, 1)

        G = np.dot(UB.T, UB)

        B = scipy.linalg.cholesky(G, lower=False)
        Bp = np.dot(B, W)

        Q, R = scipy.linalg.qr(Bp)

        self.V = np.dot(R.T, R)

    def calculate_transforms(self, signal, axes, labels, normal, value):
        ind = np.array(normal) != 1

        axes_ind = np.arange(3)[ind]
        slice_ind = np.arange(3)[~ind][0]

        coords = [axes[i] for i in axes_ind]
        titles = [labels[i] for i in axes_ind]

        z = axes[slice_ind]
        i = np.argmin(np.abs(z - value))

        titles.append(f"{labels[slice_ind]} = {z[i]:.4f}")

        if slice_ind == 0:
            data = signal[i, :, :].T
        elif slice_ind == 1:
            data = signal[:, i, :].T
        else:
            data = signal[:, :, i].T

        v = scipy.linalg.cholesky(self.V[ind][:, ind], lower=False)
        v /= v[0, 0]

        T = np.eye(3)
        T[:2, :2] = v

        s = np.diag(T).copy()
        T[1, 1] = 1
        T[0, 2] = -T[0, 1] * coords[1].min()

        return coords, data, titles, T, s[1]

    def make_slice(self, coords, data, titles, T, aspect):
        x, y = coords

        transform = Affine2D(T) + self.ax.transData

        norm = "log" if (data[np.isfinite(data)] > 0).sum() > 2 else "linear"

        im = self.ax.pcolormesh(
            x,
            y,
            data,
            norm=norm,
            shading="nearest",
            transform=transform,
            rasterized=True,
        )

        self.ax.set_aspect(aspect)
        self.ax.set_xlabel(titles[0])
        self.ax.set_ylabel(titles[1])
        self.ax.set_title(titles[2])
        self.ax.minorticks_on()

        self.ax.xaxis.get_major_locator().set_params(integer=True)
        self.ax.yaxis.get_major_locator().set_params(integer=True)

        cb = self.fig.colorbar(im, ax=self.ax)
        cb.minorticks_on()
