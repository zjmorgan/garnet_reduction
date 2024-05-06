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

    def laue_integrate(self):

        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        lamda_min, lamda_max = data.wavelength_band

        runs = self.plan['Runs']

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

                self.predict_satelite_peaks('peaks',
                                            'md_corr',
                                            lamda_min,
                                            lamda_max)

            self.peaks, self.data = peaks, data

            r_cut = self.estimate_peak_size('peaks', 'md_corr')

            self.fit_peaks('peaks', r_cut)

            peaks.combine_peaks('peaks', 'combine')

        peaks.remove_weak_peaks('combine')

        peaks.save_peaks(output_file, 'combine')

        return output_file

    def laue_combine(self, files):

        output_file = self.get_output_file()

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

        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        runs = self.plan['Runs']

        data.load_data('data',
                       self.plan['IPTS'],
                       runs,
                       self.plan.get('Grouping'))

        data.load_generate_normalization(self.plan['VanadiumFile'], 'data')

        data.convert_to_Q_sample('data', 'md_data', lorentz_corr=False)
        data.convert_to_Q_sample('data', 'md_corr', lorentz_corr=True)

        for ws in ['md_data', 'md_corr', 'norm']:
            file = output_file.replace('.nxs', '_{}.nxs'.format(ws))
            data.save_histograms(file, ws, sample_logs=True)

        mtd.clear()

        return output_file

    def monochromatic_combine(self, files):

        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan['Instrument']])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        lamda_min, lamda_max = data.wavelength_band

        for ws in ['md_data', 'md_corr', 'norm']:

            merge = []

            for file in files:
                md_file = file.replace('.nxs', '_{}.nxs'.format(ws))
                data.load_histograms(md_file, md_file)
                merge.append(md_file)
                os.remove(md_file)

            if ws == 'md_data':
    
                if self.plan.get('UBFile') is None:
                    UB_file = output_file.replace('.nxs', '.mat')
                    data.save_UB(UB_file, md_file)
                    self.plan['UBFile'] = UB_file

                for md_file in merge:      
        
                    data.load_clear_UB(self.plan['UBFile'], md_file)
        
                    peaks.predict_peaks(md_file,
                                        'peaks',
                                        self.params['Centering'],
                                        self.params['MinD'],
                                        lamda_min,
                                        lamda_max)

                    if self.params['MaxOrder'] > 0:

                        self.predict_satelite_peaks('peaks', 
                                                    md_file,
                                                    lamda_min, 
                                                    lamda_max)

                        peaks.remove_duplicate_peaks('peaks')
    
                    peaks.combine_peaks('peaks', 'combine')

            data.combine_Q_sample(merge, ws)

            if 'md' in ws:
                md_file = output_file.replace('.nxs', '_{}.nxs'.format(ws))
                data.save_histograms(md_file, ws, sample_logs=True)

        peaks.renumber_runs_by_index('md_data', 'combine')

        peaks.remove_duplicate_peaks('combine')

        self.peaks, self.data = peaks, data

        r_cut = self.estimate_peak_size('combine', 'md_corr')

        self.fit_peaks('combine', r_cut)

        peaks.remove_weak_peaks('combine')

        peaks.convert_peaks('combine')

        peaks.save_peaks(output_file, 'combine')

        opt = Optimization('combine')
        opt.optimize_lattice(self.params['Cell'])

        ub_file = os.path.splitext(output_file)[0]+'.mat'

        ub = UBModel('combine')
        ub.save_UB(ub_file)

        mtd.clear()
    
    def estimate_peak_size(self, peaks_ws, data_ws):
        """
        Integrate peaks with spherical envelope up to cutoff size.
        Estimates spherical envelope radius.

        Parameters
        ----------
        peaks_ws : str
            Reference peaks table.
        data_ws : str
            Q-sample data.

        Returns
        -------
        r_cut : float
            Update cutoff radius.

        """

        peaks = self.peaks

        r_cut = self.params['Radius']

        rad, sig_noise, intens = peaks.intensity_vs_radius(data_ws,
                                                           peaks_ws,
                                                           r_cut)

        sphere = PeakSphere(r_cut)
        r_cut = sphere.fit(rad, sig_noise)

        peaks.integrate_peaks(data_ws,
                              peaks_ws,
                              r_cut,
                              method='ellipsoid')

        peaks.remove_weak_peaks('peaks')

        rad, sig_noise, intens = peaks.intensity_vs_radius(data_ws,
                                                           peaks_ws,
                                                           r_cut)

        return r_cut

    def predict_satelite_peaks(self, peaks_ws, data_ws, lamda_min, lamda_max):
        """
        Locate satellite peaks from goniometer angles.

        Parameters
        ----------
        peaks_ws : str
            Reference peaks table.
        data_ws : str
            Q-sample data with goniometer(s).
        lamda_min : float
            Minimum wavelength.
        lamda_max : float
            Maximum wavelength.

        """

        peaks = self.peaks

        Rs = peaks.get_all_goniometer_matrices(data_ws)

        for R in Rs:

            peaks.set_goniometer(peaks_ws, R)

            peaks.predict_modulated_peaks(peaks_ws,
                                          self.params['MinD'],
                                          lamda_min,
                                          lamda_max,
                                          self.params['ModVec1'],
                                          self.params['ModVec2'],
                                          self.params['ModVec3'],
                                          self.params['MaxOrder'],
                                          self.params['CrossTerms'])
    
    def fit_peaks(self, peaks_ws, r_cut):
        """
        Integrate peaks.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        r_cut : float
            Cutoff radius.

        """

        data = self.data

        plot_path = self.get_plot_path()

        peak = PeakModel(peaks_ws)

        n = peak.get_number_peaks()

        for i in range(n):

            params = peak.get_peak_shape(i)

            bins, extents = self.bin_extent(*params, r_cut)

            d, n, Q0, Q1, Q2 = data.normalize_to_Q_sample('md_data',
                                                          extents,
                                                          bins)

            ellipsoid = PeakEllipsoid(*params, r_cut, self.params['Radius'])

            params = ellipsoid.fit(Q0, Q1, Q2, d, n)

            if params is not None:

                peak.set_peak_shape(i, *params)

                intens, sig = ellipsoid.integrate(Q0,
                                                  Q1,
                                                  Q2,
                                                  d,
                                                  n,
                                                  *params)

                c, S, W, *fitting = ellipsoid.best_fit
                vals = ellipsoid.interp_fit

                peak.set_peak_intensity(i, intens, sig)

                plot = PeakPlot(fitting)

                plot.add_ellipsoid(c, S, W, vals)

                peak_name = peak.get_peak_name(i)

                plot.save_plot(os.path.join(plot_path, peak_name+'.png'))

    def bin_extent(self, c0, c1, c2, r0, r1, r2, v0, v1, v2, r_cut):
        """
        Region extent and binning around a peak based on its initial shape.

        """

        diameter = 2*r_cut

        W = np.column_stack([v0, v1, v2])
        V = np.diag([r0**2, r1**2, r2**2])

        S = (W @ V) @ W.T

        dQ = 2*np.sqrt(np.diag(S))

        dQ0, dQ1, dQ2 = dQ

        r0 = diameter if dQ0 > diameter or np.isclose(dQ0, 0) else diameter
        r1 = diameter if dQ1 > diameter or np.isclose(dQ1, 0) else diameter
        r2 = diameter if dQ2 > diameter or np.isclose(dQ2, 0) else diameter

        bins = [29, 30, 31]
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

    def get_output_file(self):
        """
        Name of output file.

        Returns
        -------
        output_file : str
            Integration output file.

        """

        output_file = os.path.join(self.plan['OutputPath'],
                                   'integration',
                                   self.plan['OutputName']+'.nxs')

        return output_file

    def get_plot_path(self):
        """
        Plot directory.

        Returns
        -------
        plot_path : str
            Path name to save plots.

        """

        return os.path.join(self.plan['OutputPath'], 'integration', 'plots')


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

        y[y < 0] = 0

        if np.isclose(y_max, 0):
            y_max = np.inf

        self.params.add('A', value=y_max, min=0, max=100*y_max, vary=True)

        out = Minimizer(self.residual,
                        self.params,
                        fcn_args=(x, y),
                        nan_policy='omit')

        result = out.minimize(method='least_squares', loss='soft_l1')

        self.params = result.params

        return 4*result.params['sigma'].value

    def best_fit(self, r):

        A = self.params['A'].value
        sigma = self.params['sigma'].value

        return self.model(r, A, sigma), A, sigma


class PeakEllipsoid:

    def __init__(self, c0, c1, c2, r0, r1, r2, v0, v1, v2, delta, r_cut):

        params = Parameters()

        self.params = params

        phi, theta, omega = self.angles(v0, v1, v2)

        self.update_constraints(c0, c1, c2,
                                r0, r1, r2,
                                phi, theta, omega,
                                delta, r_cut)

    def profile_axis(self, Q0, rotation=False):

        if rotation:
            k = np.cross([0,1,0], Q0)
            n = k/np.linalg.norm(k)
        else:
            n = Q0/np.linalg.norm(Q0)

        return n

    def projection_axes(self, n):

        n_ind = np.argmin(np.abs(n))

        u = np.zeros(3)
        u[n_ind] = 1

        u = np.cross(n, u)
        u /= np.linalg.norm(u)

        v = np.cross(n, u)
        v *= np.sign(np.dot(np.cross(u, n), v))

        if np.abs(u[1]) > np.abs(v[1]):
            u, v = v, -u

        return u, v

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

        Q = [c0, c1, c2]
        self.n = self.profile_axis(Q)
        self.u, self.v = self.projection_axes(self.n)

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

    def residual(self, params, xye_1d, xye_2d, xye_3d):

        (Q0, Q1, Q2), y_3d, e_3d = xye_3d
        (Qu, Qv), y_2d, e_2d = xye_2d
        Q, y_1d, e_1d = xye_1d

        A_1d = params['A_1d'].value
        A_2d = params['A_2d'].value
        A_3d = params['A_3d'].value

        B_1d = params['B_1d'].value
        B_2d = params['B_2d'].value
        B_3d = params['B_3d'].value

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

        args = Q0, Q1, Q2, A_3d, B_3d, c0, c1, c2, \
               sigma0, sigma1, sigma2, phi, theta, omega

        y_3d_fit = self.func(*args)

        args = Qu, Qv, A_2d, B_2d, c0, c1, c2, \
               sigma0, sigma1, sigma2, phi, theta, omega

        y_2d_fit = self.projection(*args)

        args = Q, A_1d, B_1d, c0, c1, c2, \
              sigma0, sigma1, sigma2, phi, theta, omega

        y_1d_fit = self.profile(*args)

        # w_1d = 1/np.sqrt(np.sum(1/e_1d**2))/e_1d
        # w_2d = 1/np.sqrt(np.sum(1/e_2d**2))/e_2d
        # w_3d = 1/np.sqrt(np.sum(1/e_3d**2))/e_3d

        w_1d = 1/e_1d
        w_2d = 1/e_2d
        w_3d = 1/e_3d

        diff = np.concatenate([(y_1d-y_1d_fit)*w_1d,
                               (y_2d-y_2d_fit)*w_2d,
                               (y_3d-y_3d_fit)*w_3d])

        diff[~np.isfinite(diff)] = 1e+15

        return diff

    def func(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2,
                   sigma0, sigma1, sigma2, phi, theta, omega):

        y = self.generalized3d(Q0, Q1, Q2, mu0, mu1, mu2,
                               sigma0, sigma1, sigma2, phi, theta, omega)

        return A*y+B

    def projection(self, Qu, Qv, A, B, mu0, mu1, mu2,
                         sigma0, sigma1, sigma2, phi, theta, omega):

        c = np.array([mu0, mu1, mu2])
        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        mu_u, mu_v, sigma_u, sigma_v, rho = self.projection_params(c, S)

        y = self.generalized2d(Qu, Qv, mu_u, mu_v, sigma_u, sigma_v, rho)

        return A*y+B

    def profile(self, Q, A, B, mu0, mu1, mu2,
                      sigma0, sigma1, sigma2, phi, theta, omega):

        c = np.array([mu0, mu1, mu2])
        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        mu, sigma = self.profile_params(c, S)

        y = self.generalized1d(Q, mu, sigma)

        return A*y+B

    def profile_params(self, c, S):

        mu = np.dot(c, self.n)

        s = np.sqrt(np.dot(np.dot(S, self.n), self.n))

        return mu, s

    def projection_params(self, c, S):

        mu_u = np.dot(c, self.u)
        mu_v = np.dot(c, self.v)

        uv = [self.u, self.v]

        cov = np.einsum('ki,li->kl', np.einsum('ij,kj->ki', S, uv), uv)

        s = np.sqrt(np.diag(cov))
        corr = cov[0,1]/s[0]/s[1]

        return mu_u, mu_v, *s, corr

    def generalized3d(self, Q0, Q1, Q2, mu0, mu1, mu2,
                            sigma0, sigma1, sigma2, phi, theta, omega):

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        inv_S = self.inv_S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        dx = [x0, x1, x2]

        d = np.sqrt(np.einsum('i...,i...->...',
                    np.einsum('ij,j...->i...', inv_S, dx), dx))

        return np.exp(-0.5*d**2)#/np.sqrt(np.linalg.det(2*np.pi*S))

    def generalized2d(self, Qu, Qv, mu_u, mu_v, sigma_u, sigma_v, rho):

        xu, xv = Qu-mu_u, Qv-mu_v

        S = np.array([[sigma_u**2, sigma_u*sigma_v*rho],
                      [sigma_u*sigma_v*rho, sigma_v**2]])

        inv_S = np.linalg.inv(S)

        dx = [xu, xv]

        d = np.sqrt(np.einsum('i...,i...->...',
                    np.einsum('ij,j...->i...', inv_S, dx), dx))

        return np.exp(-0.5*d**2)#/np.sqrt(np.linalg.det(2*np.pi*S))

    def generalized1d(self, Q, mu, sigma):

        x = (Q-mu)/sigma

        return np.exp(-0.5*x**2)#/np.sqrt(2*np.pi*sigma**2)

    def voxel_volume(self, x0, x1, x2):

        return (x0[1,0,0]-x0[0,0,0])*\
               (x1[0,1,0]-x1[0,0,0])*\
               (x2[0,0,1]-x2[0,0,0])

    def bin1d(self, x0, x1, x2, d, n):

        Q = np.einsum('i,i...->...', self.n, [x0, x1, x2])
        Q_bins = np.histogram_bin_edges(Q, bins='auto')

        data_bins, _ = np.histogram(Q, bins=Q_bins, weights=d)
        norm_bins, _ = np.histogram(Q, bins=Q_bins, weights=n)

        y = data_bins/norm_bins
        e = np.sqrt(data_bins)/norm_bins
        x = (Q_bins[:-1]+Q_bins[1:])*0.5

        return x, y, e

    def bin2d(self, x0, x1, x2, d, n):

        Qu = np.einsum('i,i...->...', self.u, [x0, x1, x2])
        Qv = np.einsum('i,i...->...', self.v, [x0, x1, x2])

        Qu_bins = np.histogram_bin_edges(Qu, bins='auto')
        Qv_bins = np.histogram_bin_edges(Qv, bins='auto')

        bins = [Qu_bins, Qv_bins]

        data_bins, _, _ = np.histogram2d(Qu, Qv, bins=bins, weights=d)
        norm_bins, _, _ = np.histogram2d(Qu, Qv, bins=bins, weights=n)

        y = data_bins/norm_bins
        e = np.sqrt(data_bins)/norm_bins

        xu = (Qu_bins[:-1]+Qu_bins[1:])*0.5
        xv = (Qv_bins[:-1]+Qv_bins[1:])*0.5

        xu, xv = np.meshgrid(xu, xv, indexing='ij')

        return (xu, xv), y, e

    def fit(self, x0, x1, x2, d, n):

        mask = (d > 0) & (n > 0)

        self.params.add('A_1d', value=0, min=0, max=1, vary=False)
        self.params.add('A_2d', value=0, min=0, max=1, vary=False)
        self.params.add('A_3d', value=0, min=0, max=1, vary=False)

        self.params.add('B_1d', value=0, min=0, max=1, vary=False)
        self.params.add('B_2d', value=0, min=0, max=1, vary=False)
        self.params.add('B_3d', value=0, min=0, max=1, vary=False)

        if mask.sum() > 31:

            x = [x0[mask], x1[mask], x2[mask]]

            y = d[mask]/n[mask]
            e = np.sqrt(d[mask])/n[mask]

            xye_3d = x, y, e

            (Q0, Q1, Q2), y, e = xye_3d
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if y_max <= y_min or np.isclose(y_max, 0):
                y_max = 1

            self.params['A_3d'].set(value=y_max, min=0, max=5*y_max, vary=True)
            self.params['B_3d'].set(value=y_min, min=0, max=y_max, vary=True)

            xye_1d = self.bin1d(x0[mask], x1[mask], x2[mask], d[mask], n[mask])
            xye_2d = self.bin2d(x0[mask], x1[mask], x2[mask], d[mask], n[mask])

            Q, y, e = xye_1d

            mask = (y > 0) & (e > 0)

            Q, y, e = Q[mask], y[mask], e[mask]

            xye_1d = Q, y, e
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if y_max <= y_min or np.isclose(y_max, 0):
                y_max = 1

            self.params['A_1d'].set(value=y_max, min=0, max=5*y_max, vary=True)
            self.params['B_1d'].set(value=y_min, min=0, max=y_max, vary=True)

            (Qu, Qv), y, e = xye_2d

            mask = (y > 0) & (e > 0)

            Qu, Qv, y, e = Qu[mask], Qv[mask], y[mask], e[mask]

            xye_2d = (Qu, Qv), y, e
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            if y_max <= y_min or np.isclose(y_max, 0):
                y_max = 1

            self.params['A_2d'].set(value=y_max, min=0, max=5*y_max, vary=True)
            self.params['B_2d'].set(value=y_min, min=0, max=y_max, vary=True)

            out = Minimizer(self.residual,
                            self.params,
                            fcn_args=(xye_1d, xye_2d, xye_3d),
                            nan_policy='omit')

            result = out.minimize(method='least_squares', loss='soft_l1')

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

            out = Minimizer(self.residual,
                            self.params,
                            fcn_args=(xye_1d, xye_2d, xye_3d),
                            nan_policy='omit')

            result = out.minimize(method='least_squares', loss='soft_l1')

            self.params = result.params

            A_1d = self.params['A_1d'].value
            A_2d = self.params['A_2d'].value
            A_3d = self.params['A_3d'].value

            B_1d = self.params['B_1d'].value
            B_2d = self.params['B_2d'].value
            B_3d = self.params['B_3d'].value

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

            x = [x0, x1, x2]

            y = d/n
            e = np.sqrt(d)/n

            xye_3d = x, y, e

            (Q0, Q1, Q2), y, e = xye_3d

            mask = (d > 0) & (n > 0)

            xye_1d = self.bin1d(x0[mask], x1[mask], x2[mask], d[mask], n[mask])
            xye_2d = self.bin2d(x0[mask], x1[mask], x2[mask], d[mask], n[mask])

            args = Q0, Q1, Q2, A_3d, B_3d, c0, c1, c2, \
                   sigma0, sigma1, sigma2, phi, theta, omega

            y_3d_fit = self.func(*args)

            (Qu, Qv), y, e = xye_2d

            args = Qu, Qv, A_2d, B_2d, c0, c1, c2, \
                   sigma0, sigma1, sigma2, phi, theta, omega

            y_2d_fit = self.projection(*args)

            Q, y, e = xye_1d

            args = Q, A_1d, B_1d, c0, c1, c2, \
                   sigma0, sigma1, sigma2, phi, theta, omega

            y_1d_fit = self.profile(*args)

            fitting = xye_1d, y_1d_fit, xye_2d, y_2d_fit, xye_3d, y_3d_fit

            W = np.column_stack([self.n, self.u, self.v])

            self.best_fit = c, S, W, *fitting

            mu, sigma = self.profile_params(c, S)
            mu_u, mu_v, sigma_u, sigma_v, rho = self.projection_params(c, S)

            x = A_1d, B_1d, A_2d, B_2d, A_3d, B_3d

            self.interp_fit = mu, mu_u, mu_v, sigma, sigma_u, sigma_v, rho, *x

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