def cartesian_to_spherical(x, y, z):
    r     = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi   = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)

def mean_molecular_weight(xe):
    return 4.0 / (1.0 + 3.0*XH + 4.0*XH*xe) * MP

def gas_temperature(xe, u):
    mu = mean_molecular_weight(xe)
    return (GAMMA - 1.0) * u / (KB * 1e-6) * mu

def internal_energy_from_T(T, xe):
    mu = mean_molecular_weight(xe)
    return T * KB * 1e-6 / ((GAMMA - 1.0) * mu)

def electron_number_density(density_code, xe, h):
    density_conversion = MSUN_KG / KPC_CM**3 * 1e10 * h**2
    rho_kg_cm3 = density_code * density_conversion
    return xe * XH * rho_kg_cm3 / MP

def two_phase_ne_correction(ne, u, xe, sfr):
    # to avoid cold electrons, as they dont contribute much
    ne_corr = ne.copy()
    sf_mask = sfr > 0.0
    if not np.any(sf_mask):
        return ne_corr

    xe_sf = xe[sf_mask]
    u_sf  = u[sf_mask]
    u_h   = internal_energy_from_T(T_HOT_ISM,  xe_sf)
    u_c   = internal_energy_from_T(T_COLD_ISM, xe_sf)

    x = np.clip((u_h - u_sf) / (u_h - u_c), 0.0, 1.0)
    ne_corr[sf_mask] = ne[sf_mask] * (1.0 - x)
    return ne_corr

def half_mass_radius(coords, masses):
    r   = np.linalg.norm(coords, axis=1)
    idx = np.argsort(r)
    cum = np.cumsum(masses[idx])
    return r[idx][np.searchsorted(cum, cum[-1] / 2.0)]

def inertia_tensor(coords, masses):
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    r2 = x**2 + y**2 + z**2
    m  = masses
    I  = np.zeros((3, 3))
    I[0,0] = np.sum(m*(r2 - x*x));  I[1,1] = np.sum(m*(r2 - y*y));  I[2,2] = np.sum(m*(r2 - z*z))
    I[0,1] = I[1,0] = -np.sum(m*x*y)
    I[0,2] = I[2,0] = -np.sum(m*x*z)
    I[1,2] = I[2,1] = -np.sum(m*y*z)
    return I

def compute_galaxy_rotation_matrices(star_coords, star_masses,
                                      sfgas_coords, sfgas_masses,
                                      subhalo_pos, hmr_factor=2.0):
    """
    Compute face-on and edge-on rotation matrices from the inertia tensor.
    Uses stars + SF-gas within hmr_factor × R_half.

    Returns
    -------
    R_faceon : (3,3) ndarray – aligns disk normal with z-axis
    R_edgeon : (3,3) ndarray – tilts face-on view 90° (disk in x-z plane)
    eigvecs  : (3,3) ndarray – principal axes as columns
    """
    sc  = star_coords  - subhalo_pos
    sgc = sfgas_coords - subhalo_pos
    r_half = half_mass_radius(sc, star_masses)
    r_cut  = hmr_factor * r_half

    mask_s  = np.linalg.norm(sc,  axis=1) < r_cut
    mask_sg = np.linalg.norm(sgc, axis=1) < r_cut
    combined_coords = np.vstack([sc[mask_s], sgc[mask_sg]])
    combined_masses = np.concatenate([star_masses[mask_s], sfgas_masses[mask_sg]])

    I = inertia_tensor(combined_coords, combined_masses)
    eigvals, eigvecs = np.linalg.eigh(I)    # ascending eigenvalues
    normal = eigvecs[:, 0]                  # smallest eigenvalue = disk normal

    # face-on: align normal → z-hat
    z_hat = np.array([0., 0., 1.])
    axis  = np.cross(normal, z_hat)
    sinA  = np.linalg.norm(axis)
    cosA  = np.dot(normal, z_hat)
    if sinA < 1e-10:
        R_fo = np.eye(3)
    else:
        axis  /= sinA
        angle  = np.arctan2(sinA, cosA)
        R_fo   = Rotation.from_rotvec(angle * axis).as_matrix()

    # edge-on: tilt 90° about x-axis
    R_eo = Rotation.from_euler('x', 90, degrees=True).as_matrix() @ R_fo
    return R_fo, R_eo, eigvecs

def apply_rotation(coords, R):
    return (R @ coords.T).T

def plot_density_projection(coords_rotated, masses, h,
                             view='faceon', bins=500, lim_kpc=150,
                             title=None, ax=None, cmap='magma',
                             vmin=4, vmax=9):
    """
    2-D projected gas surface density via binned_statistic_2d.
    Colour map: log10(Sigma_gas)  [M_sun kpc^-2].

    view : 'faceon' (x-y plane) or 'edgeon' (x-z plane)
    """
    lim = lim_kpc * h                          # kpc → ckpc/h for binning
    if view == 'faceon':
        xi, yi = coords_rotated[:,0], coords_rotated[:,1]
        xlabel, ylabel = 'x [kpc]', 'y [kpc]'
    else:
        xi, yi = coords_rotated[:,0], coords_rotated[:,2]
        xlabel, ylabel = 'x [kpc]', 'z [kpc]'

    stat, xe_, ye_, _ = scipy.stats.binned_statistic_2d(
        xi, yi, masses, statistic='sum', bins=bins,
        range=[[-lim, lim], [-lim, lim]])

    dx_kpc = (2 * lim / bins) / h             # pixel size in kpc
    sigma  = stat * 1e10 / h / (dx_kpc**2)   # M_sun kpc^-2
    sigma  = np.where(sigma > 0, sigma, np.nan)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log10(sigma).T, origin='lower', cmap=cmap, aspect='equal',
                   extent=[-lim_kpc, lim_kpc, -lim_kpc, lim_kpc],
                   vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax,
                       label=r'$\log_{10}(\Sigma_{\rm gas})$ [$M_\odot$ kpc$^{-2}$]')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title or f'Gas density ({view})')
    return ax

def plot_galaxy_orientations(coords_faceon, coords_edgeon, masses, h,
                              lim_kpc=150, bins=400, suptitle=''):
    """Side-by-side face-on / edge-on density panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_density_projection(coords_faceon, masses, h, view='faceon',
                             bins=bins, lim_kpc=lim_kpc, title='Face-on', ax=axes[0])
    plot_density_projection(coords_edgeon, masses, h, view='edgeon',
                             bins=bins, lim_kpc=lim_kpc, title='Edge-on', ax=axes[1])
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig

def _tuna_can_minr(theta, phi, disk_radius, disk_height):
    """
    Minimum radial distance [same units as disk_radius/disk_height] at which
    a ray (polar angle theta, azimuth phi) exits the tuna-can cylinder.

    Cylinder:  |z| <= disk_height  AND  sqrt(x^2+y^2) <= disk_radius
    """
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    if cos_t < 1e-9:                          # equatorial ray → hits side
        return disk_radius / sin_t
    if sin_t < 1e-9:                          # polar ray → hits cap
        return disk_height / cos_t
    r_cap  = disk_height  / cos_t
    r_side = disk_radius  / sin_t
    return min(r_cap, r_side)

def construct_diskless_LOS(theta, phi, disk_radius, disk_height,
                            r_max=300.0, n_pts=500, spacing='log'):
    """
    Single LOS ray (Cartesian points) starting outside the disk cylinder.

    Parameters
    ----------
    theta       : float – polar angle (colatitude) in radians  [0, π]
    phi         : float – azimuthal angle in radians
    disk_radius : float – cylinder radius   [ckpc/h]
    disk_height : float – cylinder half-height [ckpc/h]
    r_max       : float – max radial distance  [ckpc/h]
    n_pts       : int   – number of sample points
    spacing     : 'log' or 'lin'

    Returns
    -------
    LOS : (n_pts, 3) ndarray  – Cartesian positions
    r   : (n_pts,)   ndarray  – radial distances
    """
    r_min = _tuna_can_minr(theta, phi, disk_radius, disk_height)
    r_min = max(r_min, 1e-3)
    if spacing == 'log':
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_pts)
    else:
        r = np.linspace(r_min, r_max, n_pts)
    LOS = spherical_to_cartesian(r, theta, phi)
    return LOS, r