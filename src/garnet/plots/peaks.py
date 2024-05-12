import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import scipy.special

from garnet.plots.base import BasePlot

class RadiusPlot(BasePlot):

    def __init__(self, r, y, y_fit):

        super(RadiusPlot, self).__init__()

        plt.close('all')

        self.fig, self.ax = plt.subplots(1, 
                                         1, 
                                         figsize=(6.4, 4.8),
                                         layout='constrained')

        self.add_radius_fit(r, y, y_fit)

    def add_radius_fit(self, r, y, y_fit):

        ax = self.ax

        ax.plot(r, y, 'o', color='C0')
        ax.plot(r, y_fit, '.', color='C1')
        ax.minorticks_on()
        ax.set_xlabel(r'$r$ [$\AA^{-1}$]')

    def add_sphere(self, r_cut, A, sigma):

        self.ax.axvline(x=r_cut, color='k', linestyle='--')

        xlim = list(self.ax.get_xlim())
        xlim[0] = 0

        x = np.linspace(*xlim, 256)

        z = x/sigma

        y = A*(scipy.special.erf(z/np.sqrt(2))-\
               np.sqrt(2/np.pi)*z*np.exp(-0.5*z**2))

        self.ax.plot(x, y, '-', color='C1')
        self.ax.set_ylabel(r'$I/\sigma$')

class PeakPlot(BasePlot):

    def __init__(self, fitting):

        super(PeakPlot, self).__init__()

        xye_1d, y_1d_fit, xye_2d, y_2d_fit, xye_3d, y_3d_fit = fitting

        plt.close('all')

        self.fig = plt.figure(figsize=(14.4, 4.8), layout='constrained')

        # (xu, xv), _, _ = xye_2d

        # ru = xu[-1,0]-xu[0,0]
        # rv = xv[0,-1]-xv[0,0]

        (x0, x1, x2), _, _ = xye_3d

        r0 = x0[-1,0,0]-x0[0,0,0]
        r1 = x1[0,-1,0]-x1[0,0,0]
        # r2 = x1[0,0,-1]-x1[0,0,0]

        sp = GridSpec(1, 3, figure=self.fig, width_ratios=[1,0.75,1.25])

        self.gs = []

        gs = GridSpecFromSubplotSpec(1, 1, subplot_spec=sp[0])

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(2,
                                     1,
                                     subplot_spec=sp[1])

        self.gs.append(gs)

        gs = GridSpecFromSubplotSpec(2,
                                     3,
                                     height_ratios=[1,1],
                                     width_ratios=[r0,r0,r1],
                                     subplot_spec=sp[2])

        self.gs.append(gs)

        self.add_profile_fit(xye_1d, y_1d_fit)
        self.add_projection_fit(xye_2d, y_2d_fit)
        self.add_ellipsoid_fit(xye_3d, y_3d_fit)

    def _color_limits(self, y1, y2):
        """
        Calculate color limits common for two arrays.

        Parameters
        ----------
        y1 : array-like
            Data array.
        y2 : array-like
            Data array.

        Returns
        -------
        vmin, vmax : float
            Color limits

        """

        vmin = np.nanmax([np.nanmin(y1), np.nanmin(y2)])
        vmax = np.nanmin([np.nanmax(y1), np.nanmax(y2)])

        if np.isclose(vmin, vmax) or vmin >= vmax:
            vmin, vmax = 0, 1

        return vmin, vmax

    def add_ellipsoid_fit(self, xye, y_fit):
        """
        Three-dimensional ellipsoids.

        Parameters
        ----------
        x, y, e : 3d-array
            Bins, signal, and error.
        y_fit : 3d-array
            Fitted result.

        """

        axes, y, e = xye

        x0, x1, x2 = axes

        y[np.isinf(y)] = np.nan
        y_fit[np.isnan(y)] = np.nan

        y0 = np.nansum(y, axis=0)
        y1 = np.nansum(y, axis=1)
        y2 = np.nansum(y, axis=2)

        y0_fit = np.nansum(y_fit, axis=0)
        y1_fit = np.nansum(y_fit, axis=1)
        y2_fit = np.nansum(y_fit, axis=2)

        # bkg0 = bkg[bkg.shape[0]//2,:,:]*1.0
        # bkg1 = bkg[:,bkg.shape[1]//2,:]*1.0
        # bkg2 = bkg[:,:,bkg.shape[2]//2]*1.0

        self.ellip = []

        gs = self.gs[2]

        ax = self.fig.add_subplot(gs[0,0])

        self.ellip.append(ax)

        vmin, vmax = self._color_limits(y2, y2_fit)

        ax.pcolormesh(x0[:,0,0],
                      x1[0,:,0],
                      y2.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest')

        # ax.contour(x0[:,0,0], x1[0,:,0], bkg2.T, levels=[0.5], zorder=1)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r'$Q_y$ [$\AA^{-1}$]')

        ax = self.fig.add_subplot(gs[1,0])

        self.ellip.append(ax)

        ax.pcolormesh(x0[:,0,0],
                      x1[0,:,0],
                      y2_fit.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest')

        # ax.contour(x0[:,0,0], x1[0,:,0], bkg2.T, levels=[0.5], zorder=1)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$Q_x$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$Q_y$ [$\AA^{-1}$]')

        ax = self.fig.add_subplot(gs[0,1])

        self.ellip.append(ax)

        vmin, vmax = self._color_limits(y1, y1_fit)

        ax.pcolormesh(x0[:,0,0],
                      x2[0,0,:],
                      y1.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest')

        # ax.contour(x0[:,0,0], x2[0,0,:], bkg1.T, levels=[0.5], zorder=1)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        ax = self.fig.add_subplot(gs[1,1])

        self.ellip.append(ax)

        ax.pcolormesh(x0[:,0,0],
                      x2[0,0,:],
                      y1_fit.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest')

        # ax.contour(x0[:,0,0], x2[0,0,:], bkg1.T, levels=[0.5], zorder=1)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$Q_x$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        ax = self.fig.add_subplot(gs[0,2])

        self.ellip.append(ax)

        vmin, vmax = self._color_limits(y0, y0_fit)

        ax.pcolormesh(x1[0,:,0],
                      x2[0,0,:],
                      y0.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest',
                      zorder=0)

        # ax.contour(x1[0,:,0], x2[0,0,:], bkg0.T, levels=[0.5], zorder=1)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        ax = self.fig.add_subplot(gs[1,2])

        self.ellip.append(ax)

        ax.pcolormesh(x1[0,:,0],
                      x2[0,0,:],
                      y0_fit.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest')

        # ax.contour(x1[0,:,0], x2[0,0,:], bkg0.T, levels=[0.5], zorder=1)

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$Q_y$ [$\AA^{-1}$]')
        ax.yaxis.set_ticklabels([])
        # ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')

        norm = Normalize(vmin, vmax)
        im = ScalarMappable(norm=norm)
        cb = self.fig.colorbar(im, ax=[self.ellip[-2], self.ellip[-1]])
        cb.ax.minorticks_on()

        self.gs = gs

    def add_projection_fit(self, xye, y_fit):
        """
        Two-dimensional ellipses.

        Parameters
        ----------
        x, y, e : 3d-array
            Bins, signal, and error.
        y_fit : 3d-array
            Fitted result.

        """

        (xu, xv), y, e = xye

        mask = np.isfinite(y)
        y_fit[~mask] = np.nan

        vmin, vmax = self._color_limits(y, y_fit)

        gs = self.gs[1]

        self.proj = []

        ax = self.fig.add_subplot(gs[0,0])

        self.proj.append(ax)

        ax.pcolormesh(xu[:,0],
                      xv[0,:],
                      y.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest')

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_ylabel(r'$\Delta{Q}_2$ [$\AA^{-1}$]')
        ax.xaxis.set_ticklabels([])

        ax = self.fig.add_subplot(gs[1,0])

        self.proj.append(ax)

        ax.pcolormesh(xu[:,0],
                      xv[0,:],
                      y_fit.T,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest')

        ax.minorticks_on()
        ax.set_aspect(1)
        ax.set_xlabel(r'$\Delta{Q}_1$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$\Delta{Q}_2$ [$\AA^{-1}$]')

        norm = Normalize(vmin, vmax)
        im = ScalarMappable(norm=norm)
        cb = self.fig.colorbar(im, ax=self.proj)
        cb.ax.minorticks_on()

    def add_profile_fit(self, xye, y_fit):
        """
        One-dimensional Gaussian.

        Parameters
        ----------
        x, y, e : 3d-array
            Bins, signal, and error.
        y_fit : 3d-array
            Fitted result.

        """

        x, y, e = xye

        gs = self.gs[0]

        ax = self.fig.add_subplot(gs[0,0])

        self.prof = ax

        ax.errorbar(x, y, e, fmt='o')
        ax.plot(x, y_fit, '.', color='C1')
        ax.minorticks_on()
        ax.set_xlabel(r'$\Delta{Q}$ [$\AA^{-1}$]')

    def add_ellipsoid(self, c, S, W, vals):
        """
        Draw ellipsoid envelopes.

        Parameters
        ----------
        c : 1d-array
            3 component center.
        S : 2d-array
            3x3 covariance matrix.
        W : 3x3-array
            3x3 projection matrix.
        vals : list
            Fitted parameters.

        """

        r = np.sqrt(np.diag(S))

        rho = [S[1,2]/r[1]/r[2],
               S[0,2]/r[0]/r[2],
               S[0,1]/r[0]/r[1]]

        mu, mu_u, mu_v, rad, ru, rv, corr, a, b, *_ = vals

        self.prof.axvline(x=mu-rad, color='k', linestyle='--')
        self.prof.axvline(x=mu+rad, color='k', linestyle='--')

        sigma = 0.25*rad

        x = np.linspace(*self.prof.get_xlim(), 256)
        y_fit = b+a*np.exp(-0.5*(x-mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2)

        self.prof.plot(x, y_fit, '-')

        for ax in self.proj:

            peak = Ellipse((0,0),
                           width=1,
                           height=1,
                           linestyle='-',
                           edgecolor='w',
                           facecolor='none',
                           rasterized=False)

            trans = Affine2D()

            peak.width = 2*np.sqrt(1+corr)
            peak.height = 2*np.sqrt(1-corr)

            trans.rotate_deg(45).scale(ru,rv).translate(mu_u,mu_v)

            peak.set_transform(trans+ax.transData)
            ax.add_patch(peak)

        for ax in self.ellip[0:2]:

            peak = Ellipse((0,0),
                           width=1,
                           height=1,
                           linestyle='-',
                           edgecolor='w',
                           facecolor='none',
                           rasterized=False)

            trans = Affine2D()

            peak.width = 2*np.sqrt(1+rho[2])
            peak.height = 2*np.sqrt(1-rho[2])

            trans.rotate_deg(45).scale(r[0],r[1]).translate(c[0],c[1])

            peak.set_transform(trans+ax.transData)
            ax.add_patch(peak)

        for ax in self.ellip[2:4]:

            peak = Ellipse((0,0),
                           width=1,
                           height=1,
                           linestyle='-',
                           edgecolor='w',
                           facecolor='none',
                           rasterized=False)

            trans = Affine2D()

            peak.width = 2*np.sqrt(1+rho[1])
            peak.height = 2*np.sqrt(1-rho[1])

            trans.rotate_deg(45).scale(r[0],r[2]).translate(c[0],c[2])

            peak.set_transform(trans+ax.transData)
            ax.add_patch(peak)

        for ax in self.ellip[4:6]:

            peak = Ellipse((0,0),
                           width=1,
                           height=1,
                           linestyle='-',
                           edgecolor='w',
                           facecolor='none',
                           rasterized=False)

            trans = Affine2D()

            peak.width = 2*np.sqrt(1+rho[0])
            peak.height = 2*np.sqrt(1-rho[0])

            trans.rotate_deg(45).scale(r[1],r[2]).translate(c[1],c[2])

            peak.set_transform(trans+ax.transData)
            ax.add_patch(peak)

    def _sci_notation(self, x):
        """
        Represent float in scientific notation using LaTeX.

        Parameters
        ----------
        x : float
            Value to convert.

        Returns
        -------
        s : str
            String representation in LaTeX.

        """

        if np.isfinite(x):
            exp = int(np.floor(np.log10(abs(x))))
            return '{:.2f}\\times 10^{{{}}}'.format(x / 10**exp, exp)
        else:
            return '\\infty'

    def add_peak_intensity(self, intens, sig_noise):
        """
        Add integrated intensities.

        Parameters
        ----------
        intens : list
            Intensity.
        sig_noise : list
            Signal-to-noise ratio.

        """

        I = r'$I={}$ [arb. unit]'
        I_sig = '$I/\sigma={:.1f}$'

        title = I.format(self._sci_notation(intens[0]))+' '+\
                I_sig.format(sig_noise[0])

        self.prof.set_title(title)
        self.proj[0].set_title(I.format(self._sci_notation(intens[1])))
        self.proj[1].set_title(I_sig.format(sig_noise[1]))

        self.ellip[0].set_title(I.format(self._sci_notation(intens[2])))
        self.ellip[1].set_title(I_sig.format(sig_noise[2]))

    def add_peak_info(self, wavelength, angles, gon):
        """
        Add peak information.

        Parameters
        ----------
        wavelength : float
            Wavelength.
        angles : list
            Scattering and azimuthal angles.
        gon : list
            Goniometer Euler angles.

        """

        ellip = self.ellip

        ellip[2].set_title(r'$\lambda={:.4f}$ [$\AA$]'.format(wavelength))
        ellip[3].set_title(r'$({:.1f},{:.1f},{:.1f})^\circ$'.format(*gon))
        ellip[4].set_title(r'$2\theta={:.2f}^\circ$'.format(angles[0]))
        ellip[5].set_title(r'$\phi={:.2f}^\circ$'.format(angles[1]))