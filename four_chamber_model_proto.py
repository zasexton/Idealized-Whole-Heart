"""
LV model using B-spline surfaces for epicardial and endocardial boundaries.

The LV is modeled as a surface of revolution:
- A B-spline curve defines the endocardial profile (cross-section in r-z plane)
- The epicardial surface is a parallel offset (constant wall thickness along normals)
- Both curves are revolved around the z-axis to create 3D surfaces

Constraints:
- Apex normal is (0, 0, -1): achieved by horizontal tangent at apex
- Epi and endo surfaces are parallel: epi is offset along endo normals

Dependencies:
    pip install pyvista numpy scipy

Run:
    python lv_bspline_model.py
"""

from __future__ import annotations
import numpy as np
import pyvista as pv
from typing import Tuple, List, Optional
from scipy.interpolate import BSpline

def create_bspline_curve(
        control_points: np.ndarray,
        degree: int = 3,
        num_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, BSpline, BSpline]:
    """
    Create a B-spline curve from control points.

    Args:
        control_points: (n, 2) array of [r, z] control points
        degree: B-spline degree (default 3 = cubic)
        num_samples: number of points to sample along curve

    Returns:
        points: (num_samples, 2) array of [r, z] points on curve
        tangents: (num_samples, 2) array of tangent vectors
        bspline_r: BSpline object for r coordinate
        bspline_z: BSpline object for z coordinate
    """
    n = len(control_points)

    # Ensure degree doesn't exceed n-1
    degree = min(degree, n - 1)

    # Clamped knot vector: repeats at ends for interpolation of endpoints
    num_knots = n + degree + 1

    # Clamped uniform knot vector
    knots = np.zeros(num_knots)
    knots[:degree + 1] = 0.0
    knots[-(degree + 1):] = 1.0

    # Interior knots uniformly spaced
    num_interior = num_knots - 2 * (degree + 1)
    if num_interior > 0:
        knots[degree + 1:degree + 1 + num_interior] = np.linspace(0, 1, num_interior + 2)[1:-1]

    # Create B-spline for each coordinate
    r_coords = control_points[:, 0]
    z_coords = control_points[:, 1]

    bspline_r = BSpline(knots, r_coords, degree)
    bspline_z = BSpline(knots, z_coords, degree)

    # Sample the curve
    t = np.linspace(0, 1, num_samples)
    r = bspline_r(t)
    z = bspline_z(t)

    # Compute tangents (derivatives)
    dr = bspline_r.derivative()(t)
    dz = bspline_z.derivative()(t)

    points = np.column_stack([r, z])
    tangents = np.column_stack([dr, dz])

    return points, tangents, bspline_r, bspline_z


def compute_curve_normals(tangents: np.ndarray) -> np.ndarray:
    """
    Compute outward-pointing normals for a profile curve.

    For a curve in the r-z plane that will be revolved around z-axis,
    the "outward" normal points in the +r direction (away from axis).

    Given tangent (dr, dz), the perpendicular is (-dz, dr) or (dz, -dr).
    We choose the one with positive r component (pointing outward).
    """
    normals = np.zeros_like(tangents)

    for i in range(len(tangents)):
        dr, dz = tangents[i]

        # Two perpendicular options: (-dz, dr) or (dz, -dr)
        # Choose the one pointing outward (positive r component when possible)
        n1 = np.array([-dz, dr])
        n2 = np.array([dz, -dr])

        # Normalize
        len1 = np.linalg.norm(n1)
        len2 = np.linalg.norm(n2)

        if len1 > 1e-10:
            n1 = n1 / len1
        if len2 > 1e-10:
            n2 = n2 / len2

        # Choose the one with positive r component (outward)
        # At apex (r=0), we want the normal pointing down (-z), so r=0, z=-1
        if n1[0] > n2[0]:
            normals[i] = n1
        elif n2[0] > n1[0]:
            normals[i] = n2
        else:
            # Equal r components (both zero at apex) - choose based on z
            # At apex with horizontal tangent, we want normal = (0, -1)
            if n1[1] < n2[1]:
                normals[i] = n1
            else:
                normals[i] = n2

    return normals


def offset_curve(
        points: np.ndarray,
        normals: np.ndarray,
        offset: float,
) -> np.ndarray:
    """
    Offset a curve along its normals by a constant distance.

    Args:
        points: (n, 2) curve points
        normals: (n, 2) unit normal vectors
        offset: offset distance (positive = outward)

    Returns:
        offset_points: (n, 2) offset curve points
    """
    offset_points = points + offset * normals

    # Ensure r >= 0 (can't go past the axis)
    offset_points[:, 0] = np.maximum(offset_points[:, 0], 0.0)

    return offset_points


def revolve_profile(
        profile_points: np.ndarray,
        num_theta: int = 64,
        theta_range: Tuple[float, float] = (0, 2 * np.pi),
) -> pv.PolyData:
    """
    Create a surface of revolution from a profile curve.

    Args:
        profile_points: (n, 2) array of [r, z] points defining the profile
        num_theta: number of angular divisions
        theta_range: (start, end) angles in radians

    Returns:
        PyVista PolyData mesh of the revolved surface
    """
    n_profile = len(profile_points)
    r = profile_points[:, 0]
    z = profile_points[:, 1]

    theta = np.linspace(theta_range[0], theta_range[1], num_theta, endpoint=False)

    # Generate 3D points
    points = np.zeros((n_profile, num_theta, 3))

    for i, t in enumerate(theta):
        points[:, i, 0] = r * np.cos(t)  # x
        points[:, i, 1] = r * np.sin(t)  # y
        points[:, i, 2] = z  # z

    # Flatten points
    points_flat = points.reshape(-1, 3)

    # Create faces (quads connecting adjacent profile points and theta steps)
    faces = []
    for i in range(n_profile - 1):
        for j in range(num_theta):
            j_next = (j + 1) % num_theta

            p0 = i * num_theta + j
            p1 = i * num_theta + j_next
            p2 = (i + 1) * num_theta + j_next
            p3 = (i + 1) * num_theta + j

            faces.extend([4, p0, p1, p2, p3])

    faces = np.array(faces)

    mesh = pv.PolyData(points_flat, faces)
    mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh


def rv_revolve_profile(
        profile_points: np.ndarray,
        num_theta: int = 32,
        theta_range: Tuple[float, float] = (0, np.pi),
        scale: float = 0.8,
) -> pv.PolyData:
    """
    Create a surface of revolution from a profile curve. Note that the radius of
    revolution varies with theta (RV)

    Args:
        profile_points: (n, 2) array of [r, z] points defining the profile
        num_theta: number of angular divisions
        theta_range: (start, end) angles in radians

    Returns:
        PyVista PolyData mesh of the revolved surface
    """
    n_profile = len(profile_points)
    r = profile_points[:, 0]
    z = profile_points[:, 1]

    theta = np.linspace(theta_range[0], theta_range[1], num_theta)

    # Generate 3D points
    points = np.zeros((n_profile, num_theta, 3))

    for i, t in enumerate(theta):
        points[:, i, 0] = r * (1 + scale * np.sin(t)) * np.cos(t)  # x
        points[:, i, 1] = r * (1 + scale * np.sin(t)) * np.sin(t)  # y
        points[:, i, 2] = z  # z

    # Flatten points
    points_flat = points.reshape(-1, 3)

    # Create faces (quads) WITHOUT wrap-around for partial revolution
    faces = []
    for i in range(n_profile - 1):
        for j in range(num_theta - 1):
            p0 = i * num_theta + j
            p1 = i * num_theta + (j + 1)
            p2 = (i + 1) * num_theta + (j + 1)
            p3 = (i + 1) * num_theta + j
            faces.extend([4, p0, p1, p2, p3])

    faces = np.array(faces)

    mesh = pv.PolyData(points_flat, faces)
    mesh = mesh.compute_normals(auto_orient_normals=True)

    return mesh


def create_lv_default_control_points() -> np.ndarray:
    """
    Create default control points for LV endocardial profile.

    Returns (n, 2) array of [r, z] control points.

    The profile is in the r-z plane where:
    - r >= 0 is the radial distance from the long axis
    - z is the long axis (apex at bottom, base at top)

    Control points go from apex (bottom) to base (top).
    """
    # Endocardial surface control points
    # P0 and P1 have same z to create horizontal tangent at apex
    endo = np.array([
        [0.0, -3.5],  # P0: Apex (on axis)
        [0.8, -3.5],  # P1: Same z as P0 -> horizontal tangent at apex
        [1.4, -2.0],  # P2: Lower body
        [1.6, -0.5],  # P3: Mid body (widest)
        [1.5, 0.5],  # P4: Upper body
        [1.3, 1.0],  # P5: Near base
        [0.9, 1.3],  # P6: Base opening
    ])

    return endo

def create_lv_mesh(
        endo_control_points: np.ndarray,
        wall_thickness: float = 0.5,
        degree: int = 3,
        num_profile_samples: int = 80,
        num_theta: int = 64,
) -> dict:
    """
    Create LV meshes from endocardial control points.

    The epicardial surface is computed as a parallel offset of the
    endocardial surface along its outward normals.

    Returns dict with:
        'endo_mesh': endocardial surface mesh
        'epi_mesh': epicardial surface mesh
        'endo_profile': sampled endocardial profile curve
        'epi_profile': sampled epicardial profile curve (offset)
        'endo_normals': normal vectors along endo profile
    """
    # Create B-spline curve for endocardium
    endo_profile, endo_tangents, _, _ = create_bspline_curve(
        endo_control_points, degree=degree, num_samples=num_profile_samples
    )

    # Ensure r >= 0
    endo_profile[:, 0] = np.maximum(endo_profile[:, 0], 0.0)

    # Compute outward normals
    endo_normals = compute_curve_normals(endo_tangents)

    # Create epicardial profile by offsetting along normals
    epi_profile = offset_curve(endo_profile, endo_normals, wall_thickness)

    # Create surfaces of revolution
    endo_mesh = revolve_profile(endo_profile, num_theta=num_theta)
    epi_mesh = revolve_profile(epi_profile, num_theta=num_theta)

    return {
        'endo_mesh': endo_mesh,
        'epi_mesh': epi_mesh,
        'endo_profile': endo_profile,
        'epi_profile': epi_profile,
        'endo_normals': endo_normals,
    }


def create_rv_mesh(
        endo_control_points: np.ndarray,
        wall_thickness: float = 0.25,
        degree: int = 3,
        num_profile_samples: int = 80,
        num_theta: int = 32,
) -> dict:
    """
    Create RV meshes from endocardial control points.

    The epicardial surface is computed as a parallel offset of the
    endocardial surface along its outward normals.

    Returns dict with:
        'endo_mesh': endocardial surface mesh
        'epi_mesh': epicardial surface mesh
        'endo_profile': sampled endocardial profile curve
        'epi_profile': sampled epicardial profile curve (offset)
        'endo_normals': normal vectors along endo profile
    """
    # Create LV
    lv_endo_cp = create_lv_default_control_points()
    lv_result = create_lv_mesh(lv_endo_cp)

    # Create B-spline curve for epicardium
    '''epi_profile, epi_tangents, _, _ = create_bspline_curve(
        endo_control_points, degree=degree, num_samples=num_profile_samples
    )'''

    epi_profile = lv_result['epi_profile'].copy()
    epi_tangents = np.gradient(epi_profile, axis=0)

    # Compute outward normals
    epi_normals = compute_curve_normals(epi_tangents)

    # Create endocardium profile by offsetting along normals
    endo_profile = offset_curve(epi_profile, -epi_normals, wall_thickness)

    # Create surfaces of revolution
    endo_mesh = rv_revolve_profile(endo_profile, num_theta=num_theta)
    epi_mesh = rv_revolve_profile(epi_profile, num_theta=num_theta)

    return {
        'endo_mesh': endo_mesh,
        'epi_mesh': epi_mesh,
        'endo_profile': endo_profile,
        'epi_profile': epi_profile,
        'epi_normals': epi_normals,
    }

def build_la_mesh(
    lv_result: dict,
    rv_result: dict,
    spacing: float = 0.05,  # Voxel grid spacing
    # LA shape (ellipsoid radii):
    r_scale_xy: tuple[float, float] = (0.72, 0.62),  # (x_radius, y_radius)
    r_scale_z: float = 0.78,  # z_radius
    # LA placement relative to LV base
    center_offset_xy: tuple[float, float] = (0.28, 0.62),  # LA placement relative to LV base (x, y) [R_base]
    center_offset_z: float = 0.72,
    av_plane_offset: float = 0.08,  # Define minimum for LA (along z-direction)
    wall_thickness: float = 0.16,  # Wall thickness (for inner shell)
    clearance: float = 1.25,  # Spacing between the pulmonary veins
    # Placement of tubes relative to LA centroid (x,y):
    x_side: float = 0.54,
    y_bias: float = -0.34,
) -> pv.PolyData:
    """
    Build a Left Atrium (LA) as an implicit ellipsoidal shell that is positioned
    relative to pre-existing LV/RV geometry.
    """

    lv_epi = lv_result["epi_mesh"]
    rv_epi = rv_result["epi_mesh"]

    # Determine where top of LV is:
    z_base = float(lv_epi.bounds[5])  # zmax

    # Find the radius of LA to match th half-span of the LVL
    x0, x1 = lv_epi.bounds[0:2]
    R_base = 0.5 * (x1 - x0)

    # Find the center of RV mesh:
    rv_cx = float(rv_epi.center[1])
    if rv_cx > 0:
        lv_sign = -1.0
    else:
        lv_sign = 1.0

    # Define minor axes stretch:
    ax = r_scale_xy[0] * R_base
    by = r_scale_xy[1] * R_base
    cz = r_scale_z * R_base

    # Define centroid:
    cx = center_offset_xy[0] * R_base
    cy = lv_sign * center_offset_xy[1] * R_base
    cz0 = z_base + center_offset_z * R_base

    z_av = z_base + av_plane_offset  # cushion

    # Inner ellipsoid radii (wall thickness)
    ax_i = max(ax - wall_thickness, 0)
    by_i = max(by - wall_thickness, 0)
    cz_i = max(cz - wall_thickness, 0)

    # Build a voxel grid that covers LA domain with a small margin:
    margin = 2.0 * spacing + 0.35 * R_base
    xmin = cx - ax - margin
    xmax = cx + ax + margin
    ymin = cy - by - margin
    ymax = cy + by + margin
    zmin = z_av - margin
    zmax = cz0 + cz + margin

    # Specify the length of the grid in (x,y,z) respectively:
    dims = (int(np.ceil((xmax - xmin) / spacing)) + 1, int(np.ceil((ymax - ymin) / spacing)) + 1, int(np.ceil((zmax - zmin) / spacing)) + 1)

    # Create the voxel grid:
    grid = pv.ImageData(dimensions=dims, spacing=(spacing, spacing, spacing), origin=(xmin, ymin, zmin))

    # Grid point coordinates (N x 3)
    X = grid.points[:, 0]
    Y = grid.points[:, 1]
    Z = grid.points[:, 2]


    # Ellipsoid implicit functions:
    # F = (x/a)^2 + (y/b)^2 + (z/c)^2 - 1
    xo = (X - cx) / ax
    yo = (Y - cy) / by
    zo = (Z - cz0) / cz
    F_outer = xo ** 2 + yo ** 2 + zo **2 - 1.0

    xi = (X - cx) / ax_i
    yi = (Y - cy) / by_i
    zi = (Z - cz0) / cz_i
    F_inner = xi **2 + yi **2 + zi ** 2 - 1.0

    inside_outer = F_outer <= 0.0
    inside_inner = F_inner <= 0.0

    la_shell = inside_outer & (~inside_inner)

    # Generate pulmonary vessels
    pv_outer_radius = 0.18 * R_base  # outer radius (thickness included)
    pv_wall_thickness = 0.06 * R_base  # tube wall thickness
    pv_inner_radius = max(pv_outer_radius - pv_wall_thickness, 1e-6)

    pv_length = 1.5 * R_base  # length vein extrudes from LA

    posterior_tilt_deg = 35.0  # angular orientation of vessels
    phi = posterior_tilt_deg * np.pi / 180  # convert angle in degrees to radians

    # Specify direction of veins:
    axis_R = np.array([ np.cos(phi), -np.sin(phi), 0.0], dtype=np.float32)
    axis_L = np.array([-np.cos(phi), -np.sin(phi), 0.0], dtype=np.float32)
    axis_R /= np.linalg.norm(axis_R)  # normalize to unit length
    axis_L /= np.linalg.norm(axis_L)  # normalize to unit length

    # Attachment height on LA
    z_attach = cz0 + 0.25 * cz


    # Spacing for pulmonary veins (prevent collisions)
    z_sep = clearance * (2.0 * pv_outer_radius)
    z_sep = min(z_sep, 0.45 * cz)  # constrain separation to "fit" in LA

    '''How far to place the inlets from LA center:
    #x_side = 0.75 * ax 
    #y_bias = -0.40 * by
    Now keyword argument
    '''

    # define centers of each vessel:
    attach_L_sup = (cx - x_side, cy + y_bias, z_attach + 0.5 * z_sep)
    attach_L_inf = (cx - x_side, cy + y_bias, z_attach - 0.5 * z_sep)
    attach_R_sup = (cx + x_side, cy + y_bias, z_attach + 0.5 * z_sep)
    attach_R_inf = (cx + x_side, cy + y_bias, z_attach - 0.5 * z_sep)

    # define tuples with (center, direction):
    pv_specs = [
        (attach_L_sup, axis_L),
        (attach_L_inf, axis_L),
        (attach_R_sup, axis_R),
        (attach_R_inf, axis_R),
    ]

    pv_tube_mask = np.zeros_like(la_shell, dtype=bool)  # boolean array with the same shape as the la_shell

    for (p0_tuple, axis) in pv_specs:
        p0 = np.array(p0_tuple, dtype=np.float32)

        # Segment endpoints
        p1 = p0 + pv_length * axis

        # Vector from p0 to each grid point
        vx = X - p0[0]
        vy = Y - p0[1]
        vz = Z - p0[2]

        # Project onto axis, clamp to [0, pv_length]
        w_dot_a = vx * axis[0] + vy * axis[1] + vz * axis[2]
        t = np.clip(w_dot_a / pv_length, 0.0, 1.0)

        # Closest point on the segment
        cxp = p0[0] + t * pv_length * axis[0]
        cyp = p0[1] + t * pv_length * axis[1]
        czp = p0[2] + t * pv_length * axis[2]

        # Squared distance to centerline
        dx = X - cxp
        dy = Y - cyp
        dz = Z - czp
        r2 = dx*dx + dy*dy + dz*dz

        inside_outer_cyl = (r2 <= pv_outer_radius * pv_outer_radius)
        inside_inner_cyl = (r2 <= pv_inner_radius * pv_inner_radius)

        # Tube wall occupancy
        pv_tube_mask |= (inside_outer_cyl & (~inside_inner_cyl))

    # Union veins into the LA shell occupancy
    la_shell = la_shell | pv_tube_mask

    # Store as scalar field and extract an isosurface
    # Use float scalars for contouring: 1.0 inside shell, 0.0 outside
    scalars = la_shell.astype(np.float32)
    grid.point_data["la"] = scalars

    # Extract surface with marching cubes
    # Contour at 0.5 between 0 and 1
    surf = grid.contour(isosurfaces=[0.5], scalars="la")

    # Clean and triangulate
    surf = surf.triangulate().clean(tolerance=1e-7)

    # Smoothing:
    surf = surf.smooth(n_iter=30, relaxation_factor=0.05, feature_smoothing=False, boundary_smoothing=True)

    # Recompute normals:
    surf = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)

    return surf

def build_ra_mesh(
    lv_result: dict,
    rv_result: dict,
    la_mesh: pv.PolyData | None = None,   # <-- pass your LA surface here to enforce clearance
    spacing: float = 0.05,
    # RA ellipsoid
    r_scale_xy: tuple[float, float] = (0.74, 0.64),
    r_scale_z: float = 0.82,
    # Placement
    center_offset_xy: tuple[float, float] = (0.18, 0.44),  # push RA further to the RV side vs LA
    center_offset_z: float = 0.85,
    av_plane_offset: float = 0.06,
    wall_thickness: float = 0.18,
    # Prevent RA–LA intersection
    la_clearance: float = 0.10,   # in units of R_base; increase if still intersecting
    # Vena cavae:
    vc_outer_radius: float | None = None,
    vc_wall_thickness: float | None = None,
    vc_length_sup: float = 1.7,   # in units of R_base
    vc_length_inf: float = 1.5,   # in units of R_base
    # Tilt controls for IVC
    tilt_deg_sup: float = 15.0,
    tilt_deg_inf: float = 45.0,
    # Bias both SVC and IVC in -y direction so IVC does not intersect the RV
    posterior_bias_deg: float = 18.0,
) -> pv.PolyData:
    """
    RA as an implicit ellipsoidal shell + SVC/IVC cylindrical shells.
    """
    lv_epi = lv_result["epi_mesh"]
    rv_epi = rv_result["epi_mesh"]

    z_base = float(lv_epi.bounds[5])  # zmax
    x0, x1 = lv_epi.bounds[0:2]
    R_base = 0.5 * (x1 - x0)

    # RV side sign (use RV center y)
    rv_cy = float(rv_epi.center[1])
    ra_sign_y = 1.0 if rv_cy >= 0 else -1.0

    # Ellipsoid radii
    ax = r_scale_xy[0] * R_base
    by = r_scale_xy[1] * R_base
    cz = r_scale_z * R_base

    # RA centroid (push to RV side)
    cx = center_offset_xy[0] * R_base
    cy = ra_sign_y * center_offset_xy[1] * R_base
    cz0 = z_base + center_offset_z * R_base

    z_av = z_base + av_plane_offset

    ax_i = max(ax - wall_thickness, 1e-6)
    by_i = max(by - wall_thickness, 1e-6)
    cz_i = max(cz - wall_thickness, 1e-6)

    if vc_outer_radius is None:
        vc_outer_radius = 0.20 * R_base
    if vc_wall_thickness is None:
        vc_wall_thickness = 0.06 * R_base
    vc_inner_radius = max(vc_outer_radius - vc_wall_thickness, 1e-6)

    # Grid bounds (include SVC/IVC extents)
    margin = 2.0 * spacing + 0.45 * R_base
    xmin = cx - ax - margin
    xmax = cx + ax + margin
    ymin = cy - by - margin
    ymax = cy + by + margin
    zmin = z_av - margin - vc_length_inf * R_base
    zmax = cz0 + cz + margin + vc_length_sup * R_base

    dims = (
        int(np.ceil((xmax - xmin) / spacing)) + 1,
        int(np.ceil((ymax - ymin) / spacing)) + 1,
        int(np.ceil((zmax - zmin) / spacing)) + 1,
    )
    grid = pv.ImageData(dimensions=dims, spacing=(spacing, spacing, spacing), origin=(xmin, ymin, zmin))

    X = grid.points[:, 0]
    Y = grid.points[:, 1]
    Z = grid.points[:, 2]

    # RA ellipsoidal shell
    xo = (X - cx) / ax
    yo = (Y - cy) / by
    zo = (Z - cz0) / cz
    F_outer = xo * xo + yo * yo + zo * zo - 1.0

    xi = (X - cx) / ax_i
    yi = (Y - cy) / by_i
    zi = (Z - cz0) / cz_i
    F_inner = xi * xi + yi * yi + zi * zi - 1.0

    inside_outer = (F_outer <= 0.0) & (Z >= z_av)
    inside_inner = (F_inner <= 0.0) & (Z >= z_av)
    ra_shell = inside_outer & (~inside_inner)

    # Vena cavae directions:
    def unit(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / (n if n > 0 else 1.0)

    # Posterior (-y):
    post = np.deg2rad(posterior_bias_deg)

    # SVC:
    axis_sup = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # IVC:
    ti = np.deg2rad(tilt_deg_inf)
    axis_inf = unit([np.sin(ti), -np.sin(post), -np.cos(ti)])

    # Move attachments
    # Posterior shift applies ONLY to IVC
    y_post_shift = -0.25 * by
    x_shift = 0.10 * ax

    # Attachment heights
    z_attach_sup = cz0 + 0.60 * cz
    z_attach_inf = cz0 - 0.50 * cz

    # SVC:
    p_attach_sup = np.array([cx, cy, z_attach_sup], dtype=np.float32)

    # IVC:
    p_attach_inf = np.array([cx + x_shift, cy + y_post_shift, z_attach_inf], dtype=np.float32)

    def add_tube_shell(mask: np.ndarray, p0: np.ndarray, axis: np.ndarray, length_Rb: float):
        L = float(length_Rb) * R_base
        vx = X - p0[0]
        vy = Y - p0[1]
        vz = Z - p0[2]
        w = vx * axis[0] + vy * axis[1] + vz * axis[2]
        t = np.clip(w / L, 0.0, 1.0)

        cxp = p0[0] + t * L * axis[0]
        cyp = p0[1] + t * L * axis[1]
        czp = p0[2] + t * L * axis[2]

        dx = X - cxp
        dy = Y - cyp
        dz = Z - czp
        r2 = dx * dx + dy * dy + dz * dz

        inside_outer_cyl = (r2 <= vc_outer_radius * vc_outer_radius)
        inside_inner_cyl = (r2 <= vc_inner_radius * vc_inner_radius)
        return mask | (inside_outer_cyl & (~inside_inner_cyl))

    tube_mask = np.zeros_like(ra_shell, dtype=bool)
    tube_mask = add_tube_shell(tube_mask, p_attach_sup, axis_sup, vc_length_sup)
    tube_mask = add_tube_shell(tube_mask, p_attach_inf, axis_inf, vc_length_inf)

    ra_shell = ra_shell | tube_mask

    # Carve out ra if intersecting la
    if la_mesh is not None and la_clearance > 0:
        # Distance from each voxel point to LA surface
        poly = la_mesh
        # Use pointset distance (fast enough for moderate grids)
        pts = pv.PolyData(grid.points)
        d = pts.compute_implicit_distance(poly)["implicit_distance"]  # signed distance; |d| is distance
        # Remove RA voxels that are within clearance of LA surface or inside LA (negative distance)
        carve = d <= (la_clearance * R_base)
        ra_shell = ra_shell & (~carve)

    # Extract isosurface
    grid.point_data["ra"] = ra_shell.astype(np.float32)
    surf = grid.contour(isosurfaces=[0.5], scalars="ra")
    surf = surf.triangulate().clean(tolerance=1e-7)
    surf = surf.smooth(n_iter=30, relaxation_factor=0.05, feature_smoothing=False, boundary_smoothing=True)
    surf = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)
    
    return surf

def build_pulmonary_trunk(
    rv_result: dict,
    outer_radius: float = 0.45,
    wall_thickness: float = 0.1,
    length: float = 2.2,
    bend: float = 0.9,
    n_samples: int = 160,
    n_theta: int = 48,
    z_offset: float = -0.02,
    slab_tol: float = 0.03,
    safety: float = 0.98,
    xy_bias: tuple[float, float] = (0.0, 0.45),
) -> dict:
    rv_endo = rv_result["endo_mesh"]

    # Choose base point for centerline:
    z_top = float(rv_endo.bounds[5])
    z0 = z_top + float(z_offset)

    pts = np.asarray(rv_endo.points)
    slab = pts[np.abs(pts[:, 2] - z0) < slab_tol]
    if len(slab) < 30:
        slab = pts[np.abs(pts[:, 2] - z0) < 3 * slab_tol]

    if len(slab) < 30:
        cx, cy = rv_endo.center[0], rv_endo.center[1]
        R_avail = 0.25 * max(rv_endo.length, 1e-6)
    else:
        cx, cy = slab[:, 0].mean(), slab[:, 1].mean()
        r = np.sqrt((slab[:, 0] - cx) ** 2 + (slab[:, 1] - cy) ** 2)
        R_avail = float(np.percentile(r, 80))

    allow = max((R_avail - outer_radius) * safety, 0.0)
    b = np.array([xy_bias[0], xy_bias[1]], dtype=float)
    nb = np.linalg.norm(b)
    if nb > 1e-12:
        b = b * min(1.0, allow / nb)
    else:
        b[:] = 0.0

    p0 = np.array([cx + b[0], cy + b[1], z0], dtype=float)

    # Create centerline:
    p1 = p0 + np.array([0.0, 0.0, 0.35 * length])
    p2 = p0 + np.array([0.35 * bend * length, 0.10 * bend * length, 0.75 * length])
    p3 = p0 + np.array([0.55 * bend * length, 0.15 * bend * length, 1.00 * length])

    t = np.linspace(0.0, 1.0, n_samples)
    cl = (
            (1 - t)[:, None] ** 3 * p0
            + 3 * (1 - t)[:, None] ** 2 * t[:, None] * p1
            + 3 * (1 - t)[:, None] * t[:, None] ** 2 * p2
            + t[:, None] ** 3 * p3
    )

    def unit(v):
        n = np.linalg.norm(v)
        return v / (n if n > 1e-12 else 1.0)

    Np = len(cl)
    T = np.zeros((Np, 3))
    for i in range(Np):
        if i == 0:
            T[i] = unit(cl[1] - cl[0])
        elif i == Np - 1:
            T[i] = unit(cl[-1] - cl[-2])
        else:
            T[i] = unit(cl[i + 1] - cl[i - 1])

    up = unit(np.array([0.0, 0.0, 1.0]))
    if abs(np.dot(up, T[0])) > 0.95:
        up = unit(np.array([0.0, 1.0, 0.0]))

    B = unit(np.cross(T[0], up))
    Nvec = unit(np.cross(B, T[0]))

    N_arr = np.zeros((Np, 3))
    B_arr = np.zeros((Np, 3))
    N_arr[0], B_arr[0] = Nvec, B

    for i in range(1, Np):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        c = np.dot(T[i - 1], T[i])
        if s < 1e-10:
            N_arr[i] = N_arr[i - 1]
            B_arr[i] = B_arr[i - 1]
            continue
        v = v / s
        ang = np.arctan2(s, c)

        # Rodrigues rotate previous normal around axis v
        a = N_arr[i - 1]
        N_new = a * np.cos(ang) + np.cross(v, a) * np.sin(ang) + v * np.dot(v, a) * (1 - np.cos(ang))
        N_new = unit(N_new)
        B_new = unit(np.cross(T[i], N_new))

        N_arr[i], B_arr[i] = N_new, B_new

    # Revolve about inner and outer radii:
    r_out = float(outer_radius)
    r_in = max(r_out - float(wall_thickness), 1e-6)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    outer_pts = np.zeros((Np, n_theta, 3))
    inner_pts = np.zeros((Np, n_theta, 3))

    for i in range(Np):
        ct = np.cos(theta)
        st = np.sin(theta)
        ring_dir = ct[:, None] * N_arr[i][None, :] + st[:, None] * B_arr[i][None, :]
        outer_pts[i] = cl[i][None, :] + r_out * ring_dir
        inner_pts[i] = cl[i][None, :] + r_in * ring_dir

    def skin(pts_grid):
        pts_flat = pts_grid.reshape(-1, 3)
        faces = []
        for i in range(Np - 1):
            for j in range(n_theta):
                jn = (j + 1) % n_theta
                p00 = i * n_theta + j
                p01 = i * n_theta + jn
                p11 = (i + 1) * n_theta + jn
                p10 = (i + 1) * n_theta + j
                faces.extend([4, p00, p01, p11, p10])
        return pv.PolyData(pts_flat, np.array(faces, dtype=np.int64)).triangulate().clean()

    outer = skin(outer_pts).compute_normals(auto_orient_normals=True)
    inner = skin(inner_pts).compute_normals(auto_orient_normals=True)

    inner_rev = inner.copy(deep=True)
    inner_rev.flip_faces()

    solid = outer.merge(inner_rev, merge_points=False).compute_normals(auto_orient_normals=True)

    return {
        "centerline": cl,
        "outer": outer,
        "inner": inner,
        "mesh": solid
    }

def build_aorta(
    lv_result: dict,
    outer_radius: float = 0.48,
    wall_thickness: float = 0.10,
    length: float = 2.0,
    arch_height: float = 1.4,
    arch_over: float = 2.2,
    twist_turns: float = 0.35,  # number of full rotations along length
    n_samples: int = 220,
    n_theta: int = 56,
    z_offset: float = -0.02,
    slab_tol: float = 0.03,
    safety: float = 0.98,
    xy_bias: tuple[float, float] = (-0.15, -0.35),
) -> dict:
    """
    Build aorta as a thick tube skinned around a cubic Bezier centerline + twist.

    Returns dict with:
      'centerline': (n_samples,3) ndarray
      'outer': pv.PolyData
      'inner': pv.PolyData
      'mesh': pv.PolyData  (outer + reversed inner; no end caps)
    """

    lv_endo = lv_result["endo_mesh"]

    # Choose base point for centerline:
    z_top = float(lv_endo.bounds[5])
    z0 = z_top + float(z_offset)

    pts = np.asarray(lv_endo.points)
    slab = pts[np.abs(pts[:, 2] - z0) < slab_tol]
    if len(slab) < 30:
        slab = pts[np.abs(pts[:, 2] - z0) < 3 * slab_tol]

    if len(slab) < 30:
        cx, cy = lv_endo.center[0], lv_endo.center[1]
        R_avail = 0.25 * max(lv_endo.length, 1e-6)
    else:
        cx, cy = slab[:, 0].mean(), slab[:, 1].mean()
        r = np.sqrt((slab[:, 0] - cx) ** 2 + (slab[:, 1] - cy) ** 2)
        R_avail = float(np.percentile(r, 80))

    allow = max((R_avail - outer_radius) * safety, 0.0)
    b = np.array([xy_bias[0], xy_bias[1]], dtype=float)
    nb = np.linalg.norm(b)
    if nb > 1e-12:
        b = b * min(1.0, allow / nb)
    else:
        b[:] = 0.0

    p0 = np.array([cx + b[0], cy + b[1], z0], dtype=float)

    # Aorta arch centerline
    # Scale controls with length
    p1 = p0 + np.array([0.10 * arch_over * length, 0.00 * arch_over * length, 0.55 * arch_height * length])
    p2 = p0 + np.array([-0.55 * arch_over * length, -0.45 * arch_over * length, 1.00 * arch_height * length])
    p3 = p0 + np.array([-1.00 * arch_over * length, -0.80 * arch_over * length, 0.70 * arch_height * length])

    t = np.linspace(0.0, 1.0, n_samples)
    cl = (
        (1 - t)[:, None] ** 3 * p0
        + 3 * (1 - t)[:, None] ** 2 * t[:, None] * p1
        + 3 * (1 - t)[:, None] * t[:, None] ** 2 * p2
        + t[:, None] ** 3 * p3
    )
    def unit(v):
        n = np.linalg.norm(v)
        return v / (n if n > 1e-12 else 1.0)

    Np = len(cl)
    T = np.zeros((Np, 3))
    for i in range(Np):
        if i == 0:
            T[i] = unit(cl[1] - cl[0])
        elif i == Np - 1:
            T[i] = unit(cl[-1] - cl[-2])
        else:
            T[i] = unit(cl[i + 1] - cl[i - 1])

    up = unit(np.array([0.0, 0.0, 1.0]))
    if abs(np.dot(up, T[0])) > 0.95:
        up = unit(np.array([0.0, 1.0, 0.0]))

    B = unit(np.cross(T[0], up))
    Nvec = unit(np.cross(B, T[0]))

    N_arr = np.zeros((Np, 3))
    B_arr = np.zeros((Np, 3))
    N_arr[0], B_arr[0] = Nvec, B

    for i in range(1, Np):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        c = np.dot(T[i - 1], T[i])
        if s < 1e-10:
            N_arr[i] = N_arr[i - 1]
            B_arr[i] = B_arr[i - 1]
            continue
        v = v / s
        ang = np.arctan2(s, c)

        a = N_arr[i - 1]
        N_new = a * np.cos(ang) + np.cross(v, a) * np.sin(ang) + v * np.dot(v, a) * (1 - np.cos(ang))
        N_new = unit(N_new)
        B_new = unit(np.cross(T[i], N_new))

        N_arr[i], B_arr[i] = N_new, B_new

    # Revolve about centerline
    r_out = float(outer_radius)
    r_in = max(r_out - float(wall_thickness), 1e-6)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    outer_pts = np.zeros((Np, n_theta, 3))
    inner_pts = np.zeros((Np, n_theta, 3))

    for i in range(Np):
        twist = 2 * np.pi * twist_turns * (i / max(Np - 1, 1))
        ct = np.cos(theta + twist)
        st = np.sin(theta + twist)
        ring_dir = ct[:, None] * N_arr[i][None, :] + st[:, None] * B_arr[i][None, :]
        outer_pts[i] = cl[i][None, :] + r_out * ring_dir
        inner_pts[i] = cl[i][None, :] + r_in * ring_dir

    def skin(pts_grid):
        pts_flat = pts_grid.reshape(-1, 3)
        faces = []
        for i in range(Np - 1):
            for j in range(n_theta):
                jn = (j + 1) % n_theta
                p00 = i * n_theta + j
                p01 = i * n_theta + jn
                p11 = (i + 1) * n_theta + jn
                p10 = (i + 1) * n_theta + j
                faces.extend([4, p00, p01, p11, p10])
        return pv.PolyData(pts_flat, np.array(faces, dtype=np.int64)).triangulate().clean()

    outer = skin(outer_pts).compute_normals(auto_orient_normals=True)
    inner = skin(inner_pts).compute_normals(auto_orient_normals=True)

    inner_rev = inner.copy(deep=True)
    inner_rev.flip_faces()
    solid = outer.merge(inner_rev, merge_points=False).compute_normals(auto_orient_normals=True)

    return {
        "centerline": cl,
        "outer": outer,
        "inner": inner,
        "mesh": solid
    }




def visualize_geometry():
    """
    Create and visualize both LV and RV together.
    Shows the interventricular septum relationship.
    """
    # Create LV
    lv_endo_cp = create_lv_default_control_points()
    lv_result = create_lv_mesh(lv_endo_cp)

    # Create RV
    rv_result = create_rv_mesh(lv_endo_cp)

    # Create LA
    la_mesh = build_la_mesh(lv_result, rv_result)

    # Create RA:
    ra_mesh = build_ra_mesh(lv_result, rv_result)

    # Create Pulmonary Trunk:
    pulm = build_pulmonary_trunk(rv_result)

    # Create Aorta:
    aorta = build_aorta(lv_result)

    # Visualize
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.set_background("white")

    # LV surfaces
    plotter.add_mesh(lv_result['endo_mesh'], color="firebrick", opacity=0.7)
    plotter.add_mesh(lv_result['epi_mesh'], color="pink", opacity=0.4)

    # RV surfaces
    plotter.add_mesh(rv_result['endo_mesh'], color="royalblue", opacity=0.7)
    plotter.add_mesh(rv_result['epi_mesh'], color="lightblue", opacity=0.4)

    # Show profile curves in the x-z plane
    lv_endo_3d = np.column_stack([
        lv_result['endo_profile'][:, 0],
        np.zeros(len(lv_result['endo_profile'])),
        lv_result['endo_profile'][:, 1],
    ])
    plotter.add_mesh(pv.lines_from_points(lv_endo_3d), color="darkred", line_width=4)

    rv_epi_3d = np.column_stack([
        rv_result['epi_profile'][:, 0],
        np.zeros(len(rv_result['epi_profile'])),
        rv_result['epi_profile'][:, 1],
    ])
    plotter.add_mesh(pv.lines_from_points(rv_epi_3d), color="steelblue", line_width=4)

    plotter.add_mesh(la_mesh, color="goldenrod", opacity=0.6)

    plotter.add_mesh(ra_mesh, color="darkviolet", opacity=0.6)

    plotter.add_mesh(pv.lines_from_points(pulm["centerline"]), color="darkgreen", line_width=3)
    plotter.add_mesh(pulm["mesh"], color="darkgreen", line_width=3)

    plotter.add_mesh(pv.lines_from_points(aorta["centerline"]), color="firebrick", line_width=3)
    plotter.add_mesh(aorta["mesh"], color="firebrick", line_width=3)


    # Add text annotation
    plotter.camera_position = 'iso'
    plotter.show()


if __name__ == "__main__":
    visualize_geometry()
