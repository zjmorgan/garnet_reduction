import numpy as np
import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D
# from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import scipy.linalg

from garnet.plots.base import BasePlot

class SlicePlot(BasePlot):

    def __init__(self, UB, W):
        
        self.fig, self.ax = plt.subplots(1, 1)

    def calculate_slice_transforms(signal, axes, labels, UB, W, normal, value):
    
        G = np.dot(UB.T, UB)
    
        B = scipy.linalg.cholesky(G, lower=False)
        Bp = np.dot(B, W)
    
        Q, R = scipy.linalg.qr(Bp)
    
        ind = np.array(normal) != 1
    
        axes_ind = np.arange(3)[ind]
        slice_ind = np.arange(3)[~ind][0]
    
        coords = [axes[i] for i in axes_ind]
        titles = [labels[i] for i in axes_ind]
    
        z = axes[slice_ind]
        i = np.argmin(np.abs(z-value))
    
        titles.append('{} = {:.2f}'.format(labels[slice_ind], z[i]))
    
        if slice_ind == 0:
            data = signal[i,:,:].T
        elif slice_ind == 1:
            data = signal[:,i,:].T
        else:
            data = signal[:,:,i].T
    
        v = scipy.linalg.cholesky(np.dot(R.T, R)[ind][:,ind], lower=False)
        v /= v[0,0]
    
        T = np.eye(3)
        T[:2,:2] = v
    
        s = np.diag(T).copy()
        T[1,1] = 1
        T[0,2] = -T[0,1]*coords[1].min()
    
        return coords, data, titles, T, s[1]

    def plot_slice(coords, data, T, aspect, titles):
    
        x, y = coords
        
        transform = Affine2D(T)+ax.transData
    
        im = ax.pcolormesh(x,
                           y,
                           data,
                           norm='log',
                           shading='nearest',
                           transform=transform)
    
        ax.set_aspect(aspect)
        ax.set_xlabel(titles[0])
        ax.set_ylabel(titles[1])
        ax.set_title(titles[2])
        ax.minorticks_on()
    
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
    
        cb = fig.colorbar(im, ax=ax)
        cb.minorticks_on()