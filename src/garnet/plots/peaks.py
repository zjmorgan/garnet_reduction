import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

import matplotlib.transforms

from mantid.simpleapi import (DivideMD,
                              IntegrateMDHistoWorkspace,
                              mtd)

class PeakPlot:

    def __init__(self, data_ws, norm_ws):

        signals = []

        dims = [mtd[data_ws].getDimension(i) for i in range(3)]

        axes = [np.linspace(dim.getMinimum(),
                            dim.getMaximum(),
                            dim.getNBoundaries()) for dim in dims]

        for i in range(3):

            app = '_{}'.format(i)

            for ws in [data_ws, norm_ws]:

                int_range = '{},{}'.format(axes[i][0],axes[i][-1])

                IntegrateMDHistoWorkspace(InputWorkspace=ws,
                                          P1Bin=int_range if i == 0 else None,
                                          P2Bin=int_range if i == 1 else None,
                                          P3Bin=int_range if i == 2 else None,
                                          OutputWorkspace=ws+app)

            DivideMD(LHSWorkspace=data_ws+app,
                     RHSWorkspace=norm_ws+app,
                     OutputWorkspace='ws'+app)

            signals.append(mtd['ws'+app].getSignalArray().squeeze().T)

        axes = [(ax[:-1]+ax[1:])*0.5 for ax in axes]

        int_range = '-inf,inf'

        r0, r1, r2 = [ax[-1]-ax[0] for ax in axes]

        plt.close('all')
        fig, ax = plt.subplots(2,
                               2,
                               figsize=(6.4,6.4),
                               width_ratios=[r0,r2],
                               height_ratios=[r2,r1])

        cfill = ax[1,0].pcolormesh(axes[0],
                                   axes[1],
                                   signals[2],
                                   shading='nearest')
        ax[1,0].minorticks_on()
        ax[1,0].set_aspect(1)
        ax[1,0].set_xlabel(r'$Q_x$ [$\AA^{-1}$]')
        ax[1,0].set_ylabel(r'$Q_y$ [$\AA^{-1}$]')

        cfill = ax[0,0].pcolormesh(axes[0],
                                   axes[2],
                                   signals[1],
                                   shading='nearest')
        ax[0,0].minorticks_on()
        ax[0,0].set_aspect(1)
        ax[0,0].set_ylabel(r'$Q_z$ [$\AA^{-1}$]')
        ax[0,0].sharex(ax[1,0])
        # ax[0,0].xaxis.set_ticklabels([])

        cfill = ax[1,1].pcolormesh(axes[2],
                                   axes[1],
                                   signals[0].T,
                                   shading='nearest')
        ax[1,1].minorticks_on()
        ax[1,1].set_aspect(1)
        ax[1,1].set_xlabel(r'$Q_z$ [$\AA^{-1}$]')
        ax[1,1].sharey(ax[1,0])
        # ax[1,1].yaxis.set_ticklabels([])

        cbar = fig.colorbar(cfill, ax=ax)
        cbar.ax.minorticks_on()
        cbar.ax.set_ylabel(r'Intensity')

        self.fig = fig
        self.ax = ax

    def add_profile_fit(self, x, y, e, y_fit):

        ax = self.ax

        ax[0,1].errorbar(x, y, e, fmt='o')
        ax[0,1].errorbar(x, y_fit)
        ax[0,1].minorticks_on()
        ax[0,1].set_xlabel(r'$|Q|$ [$\AA^{-1}$]')

    def add_ellipsoid(self, c, S):

        r = np.sqrt(np.diag(S))

        rho = [S[1,2]/r[1]/r[2],
               S[0,2]/r[0]/r[2],
               S[0,1]/r[0]/r[1]]

        ax = self.ax

        peak = Ellipse((0,0),
                       width=1,
                       height=1,
                       linestyle='-',
                       edgecolor='w',
                       facecolor='none',
                       rasterized=False)

        trans = matplotlib.transforms.Affine2D()

        peak.width = 2*np.sqrt(1+rho[2])
        peak.height = 2*np.sqrt(1-rho[2])

        trans.rotate_deg(45).scale(r[0],r[1]).translate(c[0],c[1])

        peak.set_transform(trans+ax[1,0].transData)
        ax[1,0].add_patch(peak)

        peak = Ellipse((0,0),
                       width=1,
                       height=1,
                       linestyle='-',
                       edgecolor='w',
                       facecolor='none',
                       rasterized=False)

        trans = matplotlib.transforms.Affine2D()

        peak.width = 2*np.sqrt(1+rho[1])
        peak.height = 2*np.sqrt(1-rho[1])

        trans.rotate_deg(45).scale(r[0],r[2]).translate(c[0],c[2])

        peak.set_transform(trans+ax[0,0].transData)
        ax[0,0].add_patch(peak)

        peak = Ellipse((0,0),
                       width=1,
                       height=1,
                       linestyle='-',
                       edgecolor='w',
                       facecolor='none',
                       rasterized=False)

        trans = matplotlib.transforms.Affine2D()

        peak.width = 2*np.sqrt(1+rho[0])
        peak.height = 2*np.sqrt(1-rho[0])

        trans.rotate_deg(45).scale(r[2],r[1]).translate(c[2],c[1])

        peak.set_transform(trans+ax[1,1].transData)
        ax[1,1].add_patch(peak)

    def save_plot(self, filename):

        self.fig.savefig(filename, bbox_inches='tight')
