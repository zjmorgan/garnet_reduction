import numpy as np
import scipy.special
import scipy.spatial.transform

from lmfit import Minimizer, Parameters

class PeakEllipsoid:

    def __init__(self, c0, c1, c2, r0, r1, r2, v0, v1, v2):

        params = Parameters()

        self.params = params

        phi, theta, omega = self.angles(v0, v1, v2)

        vol, a10, a20 = self.aspect_volume(r0, r1, r2)

        self.update_constraints(c0, c1, c2, vol, a10, a20, phi, theta, omega)

    def update_constraints(self, c0, c1, c2, vol, a10, a20,
                                 phi, theta, omega, delta=0.15):

        self.params.add('vol', value=vol, min=0.1*vol, max=10*vol)

        self.params.add('a10', value=a10, min=0.05, max=20)
        self.params.add('a20', value=a20, min=0.05, max=20)

        self.params.add('c0', value=c0, min=c0-delta, max=c0+delta, vary=True)
        self.params.add('c1', value=c1, min=c1-delta, max=c1+delta, vary=True)
        self.params.add('c2', value=c2, min=c2-delta, max=c2+delta, vary=True)

        self.params.add('phi', value=phi, min=-np.pi, max=np.pi)
        self.params.add('theta', value=theta, min=0, max=np.pi)
        self.params.add('omega', value=omega, min=-np.pi, max=np.pi)

    def aspect_volume(self, r0, r1, r2):

        vol = 4/3*np.pi*r0*r1*r2

        a10 = r1/r0
        a20 = r2/r0

        return vol, a10, a20

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

    def radii(self, vol, a10, a20):

        r0 = np.cbrt(vol*0.75/(np.pi*a10*a20))
        r1 = a10*r0
        r2 = a20*r0

        return r0, r1, r2

    def scale(self, r0, r1, r2, s=1/3):

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

    def residual(self, params, x, y, e):

        Q0, Q1, Q2 = x

        I = params['I'].value
        B = params['B'].value

        c0 = params['c0']
        c1 = params['c1']
        c2 = params['c2']

        vol = params['vol']
        a10 = params['a10']
        a20 = params['a20']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        sigma0, sigma1, sigma2 = self.scale(*self.radii(vol, a10, a20))

        args = Q0, Q1, Q2, I, B, c0, c1, c2, \
               sigma0, sigma1, sigma2, phi, theta, omega

        yfit = self.func(*args)

        diff = (y-yfit)/e

        diff[~np.isfinite(diff)] = 1e+15

        return diff

    def func(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2, 
                   sigma0, sigma1, sigma2, phi, theta, omega):

        y = self.generalized3d(Q0, Q1, Q2, mu0, mu1, mu2,
                               sigma0, sigma1, sigma2, phi, theta, omega)

        return A*y+B

    def generalized3d(self, Q0, Q1, Q2, mu0, mu1, mu2,
                            sigma0, sigma1, sigma2, phi, theta, omega):

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        inv_S = self.inv_S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)
        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        dx = [x0, x1, x2]

        d = np.sqrt(np.einsum('i...,i...->...',
                    np.einsum('ij,j...->i...', inv_S, dx), dx))

        return np.exp(-0.5*d**2)/np.sqrt(np.linalg.norm(2*np.pi*S))

    def loss(self, r):

        return np.abs(r).sum()

    def fit(self, x0, x1, x2, d, n):

        d3x = np.diff(x0, axis=0).mean()*\
              np.diff(x1, axis=1).mean()*\
              np.diff(x2, axis=2).mean()

        mask = (d > 0) & (n > 0)

        self.params.add('I', value=0, min=0, max=1, vary=False)
        self.params.add('B', value=0, min=0, max=1, vary=False)

        I, B = 0, 0

        if mask.sum() > 31:

            x = [x0[mask], x1[mask], x2[mask]]

            y = d[mask]/n[mask]
            e = np.sqrt(d[mask])/n[mask]

            y_min = np.nanmin(y)
            y_max = np.nanmax(y)

            y_int = np.nanmin(y-y_min)*d3x

            if y_int <= 0 or np.isclose(y_int, 0):
                y_int = 1

            self.params['I'].set(value=y_max, min=0, max=1000*y_int, vary=True)
            self.params['B'].set(value=y_min, min=0, max=y_max, vary=True)

            out = Minimizer(self.residual,
                            self.params,
                            fcn_args=(x, y, e),
                            reduce_fcn=self.loss,
                            nan_policy='omit')
            result = out.minimize(method='leastsq')

            self.params = result.params

            self.params['vol'].set(vary=False)
            self.params['a10'].set(vary=False)
            self.params['a20'].set(vary=False)

            self.params['phi'].set(vary=False)
            self.params['theta'].set(vary=False)
            self.params['omega'].set(vary=False)

            self.params['c0'].set(vary=False)
            self.params['c1'].set(vary=False)
            self.params['c2'].set(vary=False)

            self.params['B'].set(vary=True)

        I = self.params['I'].value
        B = self.params['B'].value

        I_err = self.params['I'].stderr
        B_err = self.params['B'].stderr

        if I_err is None:
            I_err = I
        if B_err is None:
            B_err = B

        c0 = self.params['c0'].value
        c1 = self.params['c1'].value
        c2 = self.params['c2'].value

        vol = self.params['vol'].value
        a10 = self.params['a10'].value
        a20 = self.params['a20'].value

        r0, r1, r2 = self.radii(vol, a10, a20)

        sigma0, sigma1, sigma2 = self.scale(r0, r1, r2)

        phi = self.params['phi'].value
        theta = self.params['theta'].value
        omega = self.params['omega'].value

        result = self.func(x0, x1, x2, I, B, c0, c1, c2,
                           sigma0, sigma1, sigma2, phi, theta, omega)

        U = self.U_matrix(phi, theta, omega)

        v0, v1, v2 = U.T

        if mask.sum() > 31:

            weights = y-B

            inv_S = self.inv_S_matrix(r0, r1, r2, phi, theta, omega)

            dx = [x[0]-c0, x[1]-c1, x[2]-c2]

            d = np.sqrt(np.einsum('i...,i...->...',
                        np.einsum('ij,j...->i...', inv_S, dx), dx))

            ellipsoid = d < 2

            weights = weights[ellipsoid]
            sum_weights = np.sum(weights)

            d0 = x[0][ellipsoid]
            d1 = x[1][ellipsoid]
            d2 = x[2][ellipsoid]

            c0 = np.sum(d0*weights)/sum_weights
            c1 = np.sum(d1*weights)/sum_weights
            c2 = np.sum(d2*weights)/sum_weights

            s0 = np.sum((d0-c0)**2*weights)/sum_weights
            s1 = np.sum((d1-c1)**2*weights)/sum_weights
            s2 = np.sum((d2-c2)**2*weights)/sum_weights

            s01 = np.sum((d0-c0)*(d1-c1)*weights)/sum_weights
            s02 = np.sum((d0-c0)*(d2-c2)*weights)/sum_weights
            s12 = np.sum((d1-c1)*(d2-c2)*weights)/sum_weights

            S = np.array([[s0,s01,s02],[s01,s1,s12],[s02,s12,s2]])

            if np.linalg.det(S) > 0:

                V, W = np.linalg.eig(S)

                r0, r1, r2 = 3*np.sqrt(V)

                v0, v1, v2 = W.T

        return c0, c1, c2, r0, r1, r2, v0, v1, v2, result