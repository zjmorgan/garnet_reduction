import numpy as np

from garnet.reduction.integration import PeakSphere, PeakEllipsoid

def test_sphere():

    r_cut = 0.25

    A = 1.2
    s = 0.1

    r = np.linspace(0, r_cut, 51)

    I = A*np.tanh((r/s)**3)

    sphere = PeakSphere(r_cut)

    radius = sphere.fit(r, I)

    assert np.tanh((radius/s)**3) > 0.95
    assert radius < r_cut

def test_ellipsoid():

    np.random.seed(13)

    nx, ny, nz = 41, 41, 41

    Qx_min, Qx_max = 0, 2
    Qy_min, Qy_max = -1.9, 2.1
    Qz_min, Qz_max = -3.2, 0.8

    Q0_x, Q0_y, Q0_z = 1.1, 0.1, -1.2

    sigma_x, sigma_y, sigma_z = 0.1, 0.15, 0.12
    rho_yz, rho_xz, rho_xy = 0.1, -0.1, -0.15

    a = 0.2
    b = 0.5
    c = 1.3

    sigma_yz = sigma_y*sigma_z
    sigma_xz = sigma_x*sigma_z
    sigma_xy = sigma_x*sigma_y

    cov = np.array([[sigma_x**2, rho_xy*sigma_xy, rho_xz*sigma_xz],
                    [rho_xy*sigma_xy, sigma_y**2, rho_yz*sigma_yz],
                    [rho_xz*sigma_xz, rho_yz*sigma_yz, sigma_z**2]])

    Q0 = np.array([Q0_x, Q0_y, Q0_z])

    signal = np.random.multivariate_normal(Q0, cov, size=1000000)

    data_norm, bins = np.histogramdd(signal,
                                     density=False,
                                     bins=[nx,ny,nz],
                                     range=[(Qx_min, Qx_max),
                                            (Qy_min, Qy_max),
                                            (Qz_min, Qz_max)])

    data_norm /= np.max(data_norm)
    data_norm /= np.sqrt(np.linalg.det(2*np.pi*cov))

    x_bin_edges, y_bin_edges, z_bin_edges = bins

    Qx = 0.5*(x_bin_edges[1:]+x_bin_edges[:-1])
    Qy = 0.5*(y_bin_edges[1:]+y_bin_edges[:-1])
    Qz = 0.5*(z_bin_edges[1:]+z_bin_edges[:-1])

    data = data_norm*c+b+a*(2*np.random.random(data_norm.shape)-1)
    norm = np.full_like(data, c)

    Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing='ij')

    params = 1, 0, -1, 0.5, 0.5, 0.5, [1,0,0], [0,1,0], [0,0,1]

    ellipsoid = PeakEllipsoid(*params)

    *params, _ = ellipsoid.fit(Qx, Qy, Qz, data, norm)

    intens, sig = ellipsoid.integrate(Qx, Qy, Qz, data, norm, *params)

    mu = params[0:3]
    radii = params[3:6]
    vectors = params[6:9]

    S = ellipsoid.S_matrix(*ellipsoid.scale(*radii, s=0.25),
                           *ellipsoid.angles(*vectors))

    s = np.sqrt(np.diag(S))
    sigma = np.sqrt(np.diag(cov))

    assert np.isclose(intens, 1, rtol=0.1)
    assert intens > 3*sig
    assert np.isclose(mu, Q0, atol=0.1).all()
    assert np.isclose(s, sigma, atol=0.05).all()
