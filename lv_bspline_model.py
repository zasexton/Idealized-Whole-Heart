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
    knots[:degree+1] = 0.0
    knots[-(degree+1):] = 1.0

    # Interior knots uniformly spaced
    num_interior = num_knots - 2 * (degree + 1)
    if num_interior > 0:
        knots[degree+1:degree+1+num_interior] = np.linspace(0, 1, num_interior + 2)[1:-1]

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
        points[:, i, 2] = z              # z

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


def create_lv_default_control_points() -> np.ndarray:
    """
    Create default control points for LV endocardial profile.

    Returns (n, 2) array of [r, z] control points.

    The profile is in the r-z plane where:
    - r >= 0 is the radial distance from the long axis
    - z is the long axis (apex at bottom, base at top)

    Control points go from apex (bottom) to base (top).

    IMPORTANT: First two control points have same z-value to ensure
    horizontal tangent at apex, giving surface normal (0, 0, -1).
    """
    # Endocardial surface control points
    # P0 and P1 have same z to create horizontal tangent at apex
    endo = np.array([
        [0.0,  -3.5],   # P0: Apex (on axis)
        [0.8,  -3.5],   # P1: Same z as P0 -> horizontal tangent at apex
        [1.4,  -2.0],   # P2: Lower body
        [1.6,  -0.5],   # P3: Mid body (widest)
        [1.5,   0.5],   # P4: Upper body
        [1.3,   1.0],   # P5: Near base
        [0.9,   1.3],   # P6: Base opening
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


class LVBSplineGUI:
    """Interactive GUI for LV B-spline model with parallel surfaces."""

    def __init__(
        self,
        degree: int = 3,
        num_profile_samples: int = 80,
        num_theta: int = 64,
    ):
        self.degree = degree
        self.num_profile_samples = num_profile_samples
        self.num_theta = num_theta

        # Initialize control points (endo only - epi is derived)
        self.endo_cp = create_lv_default_control_points()

        # Wall thickness (constant offset for parallel surfaces)
        self.wall_thickness = 0.5

        # State
        self.show_control_points = True
        self.show_normals = True
        self.show_endo = True
        self.show_epi = True
        self.endo_opacity = 0.7
        self.epi_opacity = 0.5

        # PyVista plotter
        pv.global_theme.allow_empty_mesh = True
        self.plotter = pv.Plotter(window_size=(1400, 900))
        self.plotter.add_axes()
        self.plotter.set_background("white")

        # Actors
        self.endo_actor = None
        self.epi_actor = None
        self.endo_cp_actor = None
        self.endo_profile_actor = None
        self.epi_profile_actor = None
        self.normals_actor = None

    def rebuild(self, reset_camera: bool = False):
        """Rebuild meshes from current control points."""

        # Enforce apex constraint: P0 and P1 must have same z
        # (P1's z follows P0's z to maintain horizontal tangent)
        self.endo_cp[1, 1] = self.endo_cp[0, 1]

        # Ensure apex is on axis
        self.endo_cp[0, 0] = 0.0

        # Create meshes
        result = create_lv_mesh(
            self.endo_cp,
            wall_thickness=self.wall_thickness,
            degree=self.degree,
            num_profile_samples=self.num_profile_samples,
            num_theta=self.num_theta,
        )

        endo_mesh = result['endo_mesh']
        epi_mesh = result['epi_mesh']
        endo_profile = result['endo_profile']
        epi_profile = result['epi_profile']
        endo_normals = result['endo_normals']

        # Update endocardial surface
        if self.show_endo:
            if self.endo_actor is None:
                self.endo_actor = self.plotter.add_mesh(
                    endo_mesh, name="endo", color="firebrick",
                    opacity=self.endo_opacity, smooth_shading=True,
                )
            else:
                self.endo_actor.mapper.SetInputData(endo_mesh)
                self.endo_actor.SetVisibility(True)
        elif self.endo_actor is not None:
            self.endo_actor.SetVisibility(False)

        # Update epicardial surface
        if self.show_epi:
            if self.epi_actor is None:
                self.epi_actor = self.plotter.add_mesh(
                    epi_mesh, name="epi", color="royalblue",
                    opacity=self.epi_opacity, smooth_shading=True,
                )
            else:
                self.epi_actor.mapper.SetInputData(epi_mesh)
                self.epi_actor.SetVisibility(True)
        elif self.epi_actor is not None:
            self.epi_actor.SetVisibility(False)

        # Update control point and profile visualization
        if self.show_control_points:
            # Endo control points (in 3D, on x-z plane)
            endo_cp_3d = np.column_stack([
                self.endo_cp[:, 0],
                np.zeros(len(self.endo_cp)),
                self.endo_cp[:, 1],
            ])
            endo_cp_mesh = pv.PolyData(endo_cp_3d)

            if self.endo_cp_actor is None:
                self.endo_cp_actor = self.plotter.add_mesh(
                    endo_cp_mesh, name="endo_cp", color="darkred",
                    point_size=12, render_points_as_spheres=True,
                )
            else:
                self.endo_cp_actor.mapper.SetInputData(endo_cp_mesh)
                self.endo_cp_actor.SetVisibility(True)

            # Endo profile curve
            endo_profile_3d = np.column_stack([
                endo_profile[:, 0],
                np.zeros(len(endo_profile)),
                endo_profile[:, 1],
            ])
            endo_line = pv.lines_from_points(endo_profile_3d)

            if self.endo_profile_actor is None:
                self.endo_profile_actor = self.plotter.add_mesh(
                    endo_line, name="endo_profile", color="red", line_width=2,
                )
            else:
                self.endo_profile_actor.mapper.SetInputData(endo_line)
                self.endo_profile_actor.SetVisibility(True)

            # Epi profile curve (offset)
            epi_profile_3d = np.column_stack([
                epi_profile[:, 0],
                np.zeros(len(epi_profile)),
                epi_profile[:, 1],
            ])
            epi_line = pv.lines_from_points(epi_profile_3d)

            if self.epi_profile_actor is None:
                self.epi_profile_actor = self.plotter.add_mesh(
                    epi_line, name="epi_profile", color="blue", line_width=2,
                )
            else:
                self.epi_profile_actor.mapper.SetInputData(epi_line)
                self.epi_profile_actor.SetVisibility(True)

        else:
            for actor in [self.endo_cp_actor, self.endo_profile_actor,
                          self.epi_profile_actor]:
                if actor is not None:
                    actor.SetVisibility(False)

        # Normal vectors visualization
        if self.show_normals and self.show_control_points:
            # Show normals at sparse points along profile
            step = max(1, len(endo_profile) // 15)
            sparse_pts = endo_profile[::step]
            sparse_normals = endo_normals[::step]

            # Convert to 3D (in x-z plane)
            pts_3d = np.column_stack([
                sparse_pts[:, 0],
                np.zeros(len(sparse_pts)),
                sparse_pts[:, 1],
            ])
            normals_3d = np.column_stack([
                sparse_normals[:, 0],
                np.zeros(len(sparse_normals)),
                sparse_normals[:, 1],
            ])

            # Create arrows
            arrows = pv.PolyData(pts_3d)
            arrows['vectors'] = normals_3d * 0.3  # scale for visibility
            arrows_glyphs = arrows.glyph(orient='vectors', scale=False, factor=0.3)

            if self.normals_actor is None:
                self.normals_actor = self.plotter.add_mesh(
                    arrows_glyphs, name="normals", color="green",
                )
            else:
                self.normals_actor.mapper.SetInputData(arrows_glyphs)
                self.normals_actor.SetVisibility(True)
        elif self.normals_actor is not None:
            self.normals_actor.SetVisibility(False)

        # Status text
        apex_normal = endo_normals[0]
        self.plotter.add_text(
            f"Wall: {self.wall_thickness:.2f} | Apex normal: ({apex_normal[0]:.3f}, {apex_normal[1]:.3f})",
            name="status", font_size=10, position="upper_right",
        )

        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def make_cp_setter(self, idx: int, coord: str):
        """Create a callback for adjusting a control point."""
        def callback(value: float):
            if coord == 'r':
                self.endo_cp[idx, 0] = value
            else:
                self.endo_cp[idx, 1] = value
            self.rebuild()
        return callback

    def set_wall_thickness(self, value: float):
        self.wall_thickness = value
        self.rebuild()

    def toggle_control_points(self, flag: bool):
        self.show_control_points = flag
        self.rebuild()

    def toggle_normals(self, flag: bool):
        self.show_normals = flag
        self.rebuild()

    def toggle_endo(self, flag: bool):
        self.show_endo = flag
        self.rebuild()

    def toggle_epi(self, flag: bool):
        self.show_epi = flag
        self.rebuild()

    def setup_sliders(self):
        """Add slider widgets for control point adjustment."""

        win_w, win_h = self.plotter.window_size

        slider_len = 0.14
        dy = 0.048
        title_height = 0.020
        label_height = 0.020
        slider_style = "modern"
        slider_width = 0.02
        tube_width = 0.01

        col_r = 0.01
        col_z = 0.17
        col_controls = 0.33

        def px(
            x_rel: float,
            y_rel: float,
            x_off: int = 0,
            y_off: int = 0,
        ) -> Tuple[int, int]:
            return int(x_rel * win_w) + x_off, int(y_rel * win_h) + y_off

        def add_slider(
            callback,
            rng,
            value: float,
            title: str,
            x: float,
            y: float,
            fmt: str = "%.2f",
        ):
            widget = self.plotter.add_slider_widget(
                callback,
                rng,
                value=value,
                title=title,
                pointa=(x, y),
                pointb=(x + slider_len, y),
                title_height=title_height,
                fmt=fmt,
                style=slider_style,
                slider_width=slider_width,
                tube_width=tube_width,
            )
            rep = widget.GetRepresentation()
            rep.SetLabelHeight(label_height)
            return widget

        # Column headers
        self.plotter.add_text(
            "Endo r",
            position=px(col_r, 0.965, 5, 0),
            font_size=10,
            name="hdr_endo_r",
        )
        self.plotter.add_text(
            "Endo z",
            position=px(col_z, 0.965, 5, 0),
            font_size=10,
            name="hdr_endo_z",
        )
        self.plotter.add_text(
            "Model / View",
            position=px(col_controls, 0.965, 5, 0),
            font_size=10,
            name="hdr_view",
        )

        # Endo control points (skip P0 r since it's fixed at 0)
        y = 0.93
        for i in range(len(self.endo_cp)):
            # r coordinate (skip P0 - apex must be on axis)
            if i > 0:
                add_slider(
                    self.make_cp_setter(i, 'r'),
                    [0.0, 3.0],
                    value=self.endo_cp[i, 0],
                    title=f"P{i} r",
                    x=col_r,
                    y=y,
                )

            # z coordinate (skip P1 - follows P0 for horizontal tangent)
            if i != 1:
                add_slider(
                    self.make_cp_setter(i, 'z'),
                    [-5.0, 3.0],
                    value=self.endo_cp[i, 1],
                    title=f"P{i} z",
                    x=col_z,
                    y=y,
                )

            y -= dy

        # Wall thickness slider
        y = 0.93
        add_slider(
            self.set_wall_thickness,
            [0.1, 1.5],
            value=self.wall_thickness,
            title="Wall thickness",
            x=col_controls,
            y=y,
        )

        # Toggle checkboxes
        y_toggle = 0.82

        self.plotter.add_checkbox_button_widget(
            self.toggle_control_points,
            value=self.show_control_points,
            position=px(col_controls, y_toggle),
            size=20, color_on="green", color_off="gray",
        )
        self.plotter.add_text(
            "Ctrl Pts",
            position=px(col_controls, y_toggle, 25, 3),
            font_size=9,
            name="lbl_cp",
        )

        y_toggle -= 0.05
        self.plotter.add_checkbox_button_widget(
            self.toggle_normals,
            value=self.show_normals,
            position=px(col_controls, y_toggle),
            size=20, color_on="green", color_off="gray",
        )
        self.plotter.add_text(
            "Normals",
            position=px(col_controls, y_toggle, 25, 3),
            font_size=9,
            name="lbl_normals",
        )

        y_toggle -= 0.05
        self.plotter.add_checkbox_button_widget(
            self.toggle_endo,
            value=self.show_endo,
            position=px(col_controls, y_toggle),
            size=20, color_on="firebrick", color_off="gray",
        )
        self.plotter.add_text(
            "Endo",
            position=px(col_controls, y_toggle, 25, 3),
            font_size=9,
            name="lbl_endo",
        )

        y_toggle -= 0.05
        self.plotter.add_checkbox_button_widget(
            self.toggle_epi,
            value=self.show_epi,
            position=px(col_controls, y_toggle),
            size=20, color_on="royalblue", color_off="gray",
        )
        self.plotter.add_text(
            "Epi",
            position=px(col_controls, y_toggle, 25, 3),
            font_size=9,
            name="lbl_epi",
        )

        # Add note about constraints
        self.plotter.add_text(
            "P0: apex (r=0 fixed)\nP1z: follows P0z (horiz tangent)",
            position=px(col_controls, 0.46),
            font_size=8,
            name="constraints_note",
        )

    def launch(self):
        """Launch the interactive GUI."""
        self.rebuild(reset_camera=True)
        self.setup_sliders()
        self.plotter.show()


def launch_lv_bspline_gui():
    """Convenience function to launch the LV B-spline GUI."""
    gui = LVBSplineGUI()
    gui.launch()


if __name__ == "__main__":
    launch_lv_bspline_gui()
