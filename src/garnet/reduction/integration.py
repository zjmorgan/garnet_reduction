import os

from mantid.simpleapi import mtd
from mantid import config
config['Q.convention'] = 'Crystallography'

import numpy as np

import scipy.special
import scipy.spatial.transform

from lmfit import Minimizer, Parameters

from garnet.reduction.data import DataModel
from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
from garnet.reduction.ub import UBModel, Optimization, lattice_group
from garnet.config.instruments import beamlines
from garnet.plots.peaks import PeakPlot

class Integration:

    def __init__(self, plan):

        self.plan = plan
        self.params = plan['Integration']

        self.validate_params()

    def validate_params(self):

        assert self.params['Cell'] in lattice_group.keys()
        assert self.params['Centering'] in centering_reflection.keys()
        assert self.params['MinD'] > 0
        assert self.params['Radius'] > 0

        if self.params.get('ModVec1') is None:
            self.params['ModVec1'] = [0,0,0]
        if self.params.get('ModVec2') is None:
            self.params['ModVec2'] = [0,0,0]
        if self.params.get('ModVec3') is None:
            self.params['ModVec3'] = [0,0,0]

        if self.params.get('MaxOrder') is None:
            self.params['MaxOrder'] = 0
        if self.params.get('CrossTerms') is None:
            self.params['CrossTerms'] = False

        assert len(self.params['ModVec1']) == 3
        assert len(self.params['ModVec2']) == 3
        assert len(self.params['ModVec3']) == 3

        assert self.params['MaxOrder'] >= 0
        assert type(self.params['CrossTerms']) is bool

    @staticmethod
    def integrate_parallel(plan, runs, proc):

        plan['Runs'] = runs
        plan['OutputName'] += '_p{}'.format(proc)

        data = DataModel(beamlines[plan['Instrument']])

        instance = Integration(plan)

        if data.laue:
            return instance.laue_integrate()
        else:
            return instance.monochromatic_integrate()

    def get_plot_path(self):
        
        return os.path.join(self.plan['OutputPath'], 'integration', 'plots')

    def laue_integrate(self):

        output_file = os.path.join(self.plan['OutputPath'],
                                   'integration',
                                   self.plan['OutputName']+'.nxs')

        data = DataModel(beamlines[self.plan['Instrument']])

        peaks = PeaksModel()

        lamda_min, lamda_max = data.wavelength_band

        runs = self.plan['Runs']

        plot_path = self.get_plot_path()

        data.load_generate_normalization(self.plan['VanadiumFile'],
                                         self.plan['FluxFile'])

        for run in runs:

            data.load_data('data', self.plan['IPTS'], run)

            data.apply_calibration('data',
                                   self.plan.get('DetectorCalibration'),
                                   self.plan.get('TubeCalibration'))

            data.crop_for_normalization('data')

            data.convert_to_Q_sample('data', 'md_data', lorentz_corr=False)
            data.convert_to_Q_sample('data', 'md_corr', lorentz_corr=True)

            data.load_clear_UB(self.plan['UBFile'], 'data')

            peaks.predict_peaks('data',
                                'peaks',
                                self.params['Centering'],
                                self.params['MinD'],
                                lamda_min,
                                lamda_max)

            if self.params['MaxOrder'] > 0:

                peaks.predict_modulated_peaks('combine',
                                              self.params['MinD'],
                                              lamda_min,
                                              lamda_max,
                                              self.params['ModVec1'],
                                              self.params['ModVec2'],
                                              self.params['ModVec3'],
                                              self.params['MaxOrder'],
                                              self.params['CrossTerms'])

            r_cut = self.params['Radius']

            rad, sig_noise, intens = peaks.intensity_vs_radius('md_corr',
                                                               'peaks',
                                                               r_cut)

            sphere = PeakSphere(r_cut)
            r_cut = sphere.fit(rad, sig_noise)

            peaks.integrate_peaks('md_corr',
                                  'peaks',
                                  r_cut,
                                  method='ellipsoid')

            peaks.remove_weak_peaks('peaks')

            rad, sig_noise, intens = peaks.intensity_vs_radius('md_corr',
                                                               'peaks',
                                                               r_cut)

            peak = PeakModel('peaks')

            n = peak.get_number_peaks()

            for i in range(n):

                params = peak.get_peak_shape(i)

                bins, extents = self.bin_extent(*params, r_cut)

                d, n, Q0, Q1, Q2 = data.normalize_to_Q_sample('md_data',
                                                              extents,
                                                              bins)

                ellipsoid = PeakEllipsoid(*params, r_cut, r_cut)

                params = ellipsoid.fit(Q0, Q1, Q2, d, n)

                peak.set_peak_shape(i, *params)

                intens, sig = ellipsoid.integrate(Q0,
                                                  Q1,
                                                  Q2,
                                                  d,
                                                  n,
                                                  *params)

                c, S, x, y, e, y_fit = ellipsoid.best_fit
         
                peak.set_peak_intensity(i, intens, sig)
         
                plot = PeakPlot('md_data_data', 'md_data_norm')
         
                plot.add_ellipsoid(c, S)
                plot.add_profile_fit(x, y, e, y_fit)

                peak_name = peak.get_peak_name(i)

                plot.save_plot(os.path.join(plot_path, peak_name+'.png'))

            peaks.combine_peaks('peaks', 'combine')

        peaks.remove_weak_peaks('combine')

        peaks.save_peaks(output_file, 'combine')

        return output_file

    def laue_combine(self, files):

        output_file = os.path.join(self.plan['OutputPath'],
                                   'integration',
                                   self.plan['OutputName']+'.nxs')

        peaks = PeaksModel()

        for file in files:

            peaks.load_peaks(file, 'tmp')
            peaks.combine_peaks('tmp', 'combine')
            os.remove(file)

        if mtd.doesExist('combine'):

            peaks.save_peaks(output_file, 'combine')

            opt = Optimization('combine')
            opt.optimize_lattice(self.params['Cell'])

            ub_file = os.path.splitext(output_file)[0]+'.mat'

            ub = UBModel('combine')
            ub.save_UB(ub_file)

    def monochromatic_integrate(self):

        output_file = os.path.join(self.plan['OutputPath'],
                                   'integration',
                                   self.plan['OutputName']+'.nxs')

        data = DataModel(beamlines[self.plan['Instrument']])

        runs = self.plan['Runs']

        data.load_data('data', self.plan['IPTS'], runs)

        data.load_generate_normalization(self.plan['VanadiumFile'], 'data')

        data.convert_to_Q_sample('data', 'md_data', lorentz_corr=False)
        data.convert_to_Q_sample('data', 'md_corr', lorentz_corr=True)

        data.load_clear_UB(self.plan['UBFile'], 'data')

        for ws in ['md_data', 'md_corr', 'norm']:
            file = output_file.replace('.nxs', '_{}.nxs'.format(ws))
            data.save_histograms(file, ws, sample_logs=True)

        mtd.clear()

        return output_file

    def monochromatic_combine(self, files):

        output_file = os.path.join(self.plan['OutputPath'],
                                   'integration',
                                   self.plan['OutputName']+'.nxs')

        data = DataModel(beamlines[self.plan['Instrument']])

        for ws in ['md_data', 'md_corr', 'norm']:
            merge = []
            for file in files:
                md_file = file.replace('.nxs', '_{}.nxs'.format(ws))
                data.load_histograms(md_file, md_file)
                merge.append(md_file)
                os.remove(md_file)
            data.combine_Q_sample(merge, ws)
            md_file = output_file.replace('.nxs', '_{}.nxs'.format(ws))
            data.save_histograms(md_file, ws, sample_logs=True)

        peaks = PeaksModel()

        plot_path = self.get_plot_path()

        lamda_min, lamda_max = data.wavelength_band

        peaks.predict_peaks('md_data',
                            'combine',
                            self.params['Centering'],
                            self.params['MinD'],
                            lamda_min,
                            lamda_max)

        if self.params['MaxOrder'] > 0:

            Rs = peaks.get_all_goniometer_matrices('md_data')

            for R in Rs:

                peaks.set_goniometer('combine', R)

                peaks.predict_modulated_peaks('combine',
                                              self.params['MinD'],
                                              lamda_min,
                                              lamda_max,
                                              self.params['ModVec1'],
                                              self.params['ModVec2'],
                                              self.params['ModVec3'],
                                              self.params['MaxOrder'],
                                              self.params['CrossTerms'])

        peaks.renumber_runs_by_index('md_data', 'combine')

        peaks.remove_duplicate_peaks('combine')

        r_cut = self.params['Radius']

        rad, sig_noise, intens = peaks.intensity_vs_radius('md_corr',
                                                           'combine',
                                                           r_cut)

        sphere = PeakSphere(r_cut)
        r_cut = sphere.fit(rad, sig_noise)

        peaks.integrate_peaks('md_corr',
                              'combine',
                              r_cut,
                              method='ellipsoid')

        # peaks.remove_weak_peaks('combine')

        rad, sig_noise, intens = peaks.intensity_vs_radius('md_corr',
                                                           'combine',
                                                           r_cut)

        peak = PeakModel('combine')

        n = peak.get_number_peaks()

        for i in range(n):

            params = peak.get_peak_shape(i)

            bins, extents = self.bin_extent(*params, r_cut)

            d, n, Q0, Q1, Q2 = data.normalize_to_Q_sample('md_data',
                                                          extents,
                                                          bins)

            ellipsoid = PeakEllipsoid(*params, r_cut, r_cut)

            params = ellipsoid.fit(Q0, Q1, Q2, d, n)

            peak.set_peak_shape(i, *params)

            intens, sig = ellipsoid.integrate(Q0,
                                              Q1,
                                              Q2,
                                              d,
                                              n,
                                              *params)

            c, S, x, y, e, y_fit = ellipsoid.best_fit

            peak.set_peak_intensity(i, intens, sig)

            plot = PeakPlot('md_data_data', 'md_data_norm')

            plot.add_ellipsoid(c, S)
            plot.add_profile_fit(x, y, e, y_fit)

            peak_name = peak.get_peak_name(i)

            plot.save_plot(os.path.join(plot_path, peak_name+'.png'))

        peaks.remove_weak_peaks('combine')

        peaks.save_peaks(output_file, 'combine')
 
        opt = Optimization('combine')
        opt.optimize_lattice(self.params['Cell'])

        ub_file = os.path.splitext(output_file)[0]+'.mat'

        ub = UBModel('combine')
        ub.save_UB(ub_file)

        mtd.clear()

    def bin_extent(self, c0, c1, c2, r0, r1, r2, v0, v1, v2, r_cut):
        """
        Region extent and binning around a peak based on its initial shape.

        """

        r0 = r_cut if r0 > r_cut or np.isclose(r0, 0) else r0
        r1 = r_cut if r1 > r_cut or np.isclose(r1, 0) else r1
        r2 = r_cut if r2 > r_cut or np.isclose(r2, 0) else r2

        W = np.column_stack([v0, v1, v2])
        V = np.diag([r0**2, r1**2, r2**2])

        S = (W @ V) @ W.T

        dQ = 2*np.sqrt(np.diag(S))

        dQ0, dQ1, dQ2 = dQ

        bins = [39, 40, 41]
        extents = [[c0-dQ0, c0+dQ0],
                   [c1-dQ1, c1+dQ1],
                   [c2-dQ2, c2+dQ2]]

        return bins, extents

    @staticmethod
    def combine_parallel(plan, files):

        instance = Integration(plan)

        data = DataModel(beamlines[plan['Instrument']])

        instance = Integration(plan)

        if data.laue:
            return instance.laue_combine(files)
        else:
            return instance.monochromatic_combine(files)

class PeakSphere:

    def __init__(self, r_cut):

        self.params = Parameters()

        self.params.add('sigma', value=r_cut/6, min=0.01, max=r_cut/3)

    def model(self, x, A, sigma):

        z = x/sigma

        return A*(scipy.special.erf(z/np.sqrt(2))-\
                  np.sqrt(2/np.pi)*z*np.exp(-0.5*z**2))

    def residual(self, params, x, y):

        A = params['A']
        sigma = params['sigma']

        y_fit = self.model(x, A, sigma)

        return y_fit-y

    def fit(self, x, y):

        y_max = np.max(y)

        if np.isclose(y_max, 0):
            y_max = np.inf

        self.params.add('A', value=y_max, min=0, max=100*y_max, vary=True)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(x, y),
                        reduce_fcn='negentropy',
                        nan_policy='omit')

        result = out.minimize(method='leastsq')

        return 4*result.params['sigma'].value

class PeakEllipsoid:

    def __init__(self, c0, c1, c2, r0, r1, r2, v0, v1, v2, delta, r_cut):

        params = Parameters()

        self.params = params

        phi, theta, omega = self.angles(v0, v1, v2)

        self.update_constraints(c0, c1, c2,
                                r0, r1, r2,
                                phi, theta, omega,
                                delta, r_cut)

    def update_constraints(self, c0, c1, c2, r0, r1, r2,
                                 phi, theta, omega, delta, r_cut):

        self.params.add('r0', value=r0, min=0.05*r0, max=r_cut)
        self.params.add('r1', value=r1, min=0.05*r1, max=r_cut)
        self.params.add('r2', value=r2, min=0.05*r2, max=r_cut)

        self.params.add('c0', value=c0, min=c0-delta, max=c0+delta, vary=True)
        self.params.add('c1', value=c1, min=c1-delta, max=c1+delta, vary=True)
        self.params.add('c2', value=c2, min=c2-delta, max=c2+delta, vary=True)

        self.params.add('phi', value=phi, min=-np.pi, max=np.pi)
        self.params.add('theta', value=theta, min=0, max=np.pi)
        self.params.add('omega', value=omega, min=-np.pi, max=np.pi)

        Q = np.array([c0, c1, c2])   
        self.Q = np.linalg.norm(Q)
        self.Q_hat = Q/self.Q
        self.r_cut = r_cut

    def eigenvectors(self, W):

        w = scipy.spatial.transform.Rotation.from_matrix(W).as_rotvec()

        omega = np.linalg.norm(w)

        u0, u1, u2 = (0, 0, 1) if np.isclose(omega, 0) else w/omega

        return u0, u1, u2, omega

    def angles(self, v0, v1, v2):

        W = np.column_stack([v0, v1, v2])

        u0, u1, u2, omega = self.eigenvectors(W)

        theta = np.arccos(u2)
        phi = np.arctan2(u1, u0)

        return phi, theta, omega

    def scale(self, r0, r1, r2, s=1/4):

        return s*r0, s*r1, s*r2

    def S_matrix(self, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        U = self.U_matrix(phi, theta, omega)
        V = np.diag([sigma0**2, sigma1**2, sigma2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        U = self.U_matrix(phi, theta, omega)
        V = np.diag([1/sigma0**2, 1/sigma1**2, 1/sigma2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, phi, theta, omega):

        u0 = np.cos(phi)*np.sin(theta)
        u1 = np.sin(phi)*np.sin(theta)
        u2 = np.cos(theta)

        w = omega*np.array([u0,u1,u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def residual(self, params, x, y, e, Q, Y, E):

        Q0, Q1, Q2 = x

        H = params['H'].value
        A = params['A'].value
        B = params['B'].value
        C = params['C'].value

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        r0 = params['r0']
        r1 = params['r1']
        r2 = params['r2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        sigma0, sigma1, sigma2 = self.scale(r0, r1, r2)

        args = Q0, Q1, Q2, H, B, c0, c1, c2, \
               sigma0, sigma1, sigma2, phi, theta, omega

        y_fit = self.func(*args)

        args = Q, A, C, c0, c1, c2, sigma0, sigma1, sigma2, phi, theta, omega

        Y_fit = self.profile(*args)

        f = np.sqrt(np.sum(1/e**2))
        F = np.sqrt(np.sum(1/E**2))

        diff = np.concatenate([(y-y_fit)/e/f, (Y-Y_fit)/E/F])

        diff[~np.isfinite(diff)] = 1e+15

        return diff

    def func(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2,
                   sigma0, sigma1, sigma2, phi, theta, omega):

        y = self.generalized3d(Q0, Q1, Q2, mu0, mu1, mu2,
                               sigma0, sigma1, sigma2, phi, theta, omega)

        return A*y+B

    def profile(self, Q, A, B, mu0, mu1, mu2,
                      sigma0, sigma1, sigma2, phi, theta, omega):

        mu = mu0*self.Q_hat[0]+mu1*self.Q_hat[1]+mu2*self.Q_hat[2]

        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        sigma = np.sqrt(np.dot(np.dot(S, self.Q_hat), self.Q_hat))

        y = self.generalized1d(Q, mu, sigma)

        return A*y+B

    def generalized3d(self, Q0, Q1, Q2, mu0, mu1, mu2,
                            sigma0, sigma1, sigma2, phi, theta, omega):

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        inv_S = self.inv_S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        dx = [x0, x1, x2]

        d = np.sqrt(np.einsum('i...,i...->...',
                    np.einsum('ij,j...->i...', inv_S, dx), dx))

        return np.exp(-0.5*d**2)#/np.sqrt(np.linalg.det(2*np.pi*S))

    def generalized1d(self, Q, mu, sigma):

        x = (Q-mu)/sigma

        return np.exp(-0.5*x**2)#/np.sqrt(2*np.pi*sigma**2)

    def loss(self, r):

        return np.abs(r).sum()

    def voxel_volume(self, x0, x1, x2):

        return np.diff(x0, axis=0).mean()*\
               np.diff(x1, axis=1).mean()*\
               np.diff(x2, axis=2).mean()

    def bin_in_Q(self, x0, x1, x2, d, n):

        Q = np.einsum('i,i...->...', self.Q_hat, [x0, x1, x2])
        Q_bins = np.histogram_bin_edges(Q, bins='auto')

        data_bins, _ = np.histogram(Q, bins=Q_bins, weights=d)
        norm_bins, _ = np.histogram(Q, bins=Q_bins, weights=n)

        y = data_bins/norm_bins
        e = np.sqrt(data_bins)/norm_bins
        x = (Q_bins[:-1]+Q_bins[1:])*0.5

        return x, y, e

    def fit(self, x0, x1, x2, d, n):

        mask = (d > 0) & (n > 0)

        self.params.add('H', value=0, min=0, max=1, vary=False)
        self.params.add('A', value=0, min=0, max=1, vary=False)
        self.params.add('B', value=0, min=0, max=1, vary=False)
        self.params.add('C', value=0, min=0, max=1, vary=False)

        A, C = 0, 0

        Q = np.linspace(self.Q-self.r_cut, self.Q+self.r_cut)
        Y = np.zeros_like(Q)
        E = np.zeros_like(Q)

        if mask.sum() > 31:

            x = [x0[mask], x1[mask], x2[mask]]

            y = d[mask]/n[mask]
            e = np.sqrt(d[mask])/n[mask]

            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if y_max <= y_min or np.isclose(y_max, 0):
                y_max = 1

            self.params['H'].set(value=y_max, min=0, max=1e3*y_max, vary=True)
            self.params['B'].set(value=y_min, min=0, max=y_max, vary=True)

            Q, Y, E = self.bin_in_Q(x0[mask],
                                    x1[mask],
                                    x2[mask],
                                    d[mask],
                                    n[mask])

            Y_min = np.nanmin(Y)
            Y_max = np.nanmax(Y)

            if Y_max <= Y_min or np.isclose(Y_max, 0):
                Y_max = 1

            self.params['A'].set(value=Y_max, min=0, max=1e3*Y_max, vary=True)
            self.params['C'].set(value=Y_min, min=0, max=Y_max, vary=True)

            out = Minimizer(self.residual,
                            self.params,
                            fcn_args=(x, y, e, Q, Y, E),
                            reduce_fcn=self.loss,
                            nan_policy='omit')

            result = out.minimize(method='leastsq')

            self.params = result.params

            self.params['r0'].set(vary=False)
            self.params['r1'].set(vary=False)
            self.params['r2'].set(vary=False)

            self.params['phi'].set(vary=False)
            self.params['theta'].set(vary=False)
            self.params['omega'].set(vary=False)

            self.params['c0'].set(vary=False)
            self.params['c1'].set(vary=False)
            self.params['c2'].set(vary=False)

            self.params['B'].set(vary=True)
            self.params['C'].set(vary=True)

            out = Minimizer(self.residual,
                            self.params,
                            fcn_args=(x, y, e, Q, Y, E),
                            reduce_fcn=self.loss,
                            nan_policy='omit')

            result = out.minimize(method='leastsq')

            self.params = result.params

        A = self.params['A'].value
        C = self.params['C'].value

        c0 = self.params['c0'].value
        c1 = self.params['c1'].value
        c2 = self.params['c2'].value

        r0 = self.params['r0'].value
        r1 = self.params['r1'].value
        r2 = self.params['r2'].value

        sigma0, sigma1, sigma2 = self.scale(r0, r1, r2)

        phi = self.params['phi'].value
        theta = self.params['theta'].value
        omega = self.params['omega'].value

        c = [c0, c1, c2]
        S = self.S_matrix(r0, r1, r2, phi, theta, omega)

        args = Q, A, C, c0, c1, c2, sigma0, sigma1, sigma2, phi, theta, omega

        Y_fit = self.profile(*args)

        self.best_fit = c, S, Q, Y, E, Y_fit

        U = self.U_matrix(phi, theta, omega)

        v0, v1, v2 = U.T

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def integrate(self, x0, x1, x2, d, n, c0, c1, c2, r0, r1, r2, v0, v1, v2):

        d3x = self.voxel_volume(x0, x1, x2)

        W = np.column_stack([v0, v1, v2])
        V = np.diag([1/r0**2, 1/r1**2, 1/r2**2])

        A = (W @ V) @ W.T

        dx = [x0-c0, x1-c1, x2-c2]

        dist = np.einsum('iklm,iklm->klm',
                         np.einsum('ij,jklm->iklm', A, dx), dx)

        pk = dist < 1

        struct = scipy.ndimage.generate_binary_structure(3, 1)
        dilate = scipy.ndimage.binary_dilation(pk, struct, border_value=0)

        bkg = (dilate ^ pk) & (d > 0)
        pk = (dist < 1) & (d > 0) & (n > 0)

        B = np.sum(d[bkg]/d[bkg])/np.sum(1/d[bkg])
        B_err = np.sqrt(1/np.sum(1/d[bkg]))

        num = np.nansum(d[pk]-B)
        den = np.nansum(n[pk])
        var = np.nansum(d[pk]+B_err**2)

        scale = np.sum(pk)*d3x

        intens = (num/den)*scale
        sig = np.sqrt(var)/den*scale

        return intens, sig